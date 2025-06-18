import numpy as np
import torch
import argparse
import mahotas as mh
import os

from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from functools import lru_cache
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, NamedTuple, Any, Optional

from aggrigator.uncertainty_maps import UncertaintyMap
from datasets.LIDC.lidc_dataset_creation import LIDC_UQ_Dataset, OptimizedLIDCDataset
from datasets.Arctique.arctique_dataset_creation import (
    renderHE_UQ_HVNext, 
    OptimizedArctiqueDataset, 
    inst_to_3c,
    SharedMaskCache
)
from datasets.Lizard.lizard_dataset_creation import LizardDataset
from datasets.GTA_CityScapes.gta_cityscapes_dataset_creation import GTA_CityscapesDataset
from evaluation.constants import BACKGROUND_FREE_STRATEGIES, AUROC_STRATEGIES

# ---- Data Structures ----

@dataclass
class DataPaths:
    """Container for all data paths used in the program."""
    uq_maps: Path
    metadata: Path
    predictions: Path
    data: Path
    metrics: Path
    output: Path

class AnalysisResults(NamedTuple):
    """Container for AURC analysis results."""
    mean_aurc: np.ndarray
    coverages: np.ndarray
    mean_selective_risks: np.ndarray
    std_selective_risks: np.ndarray

def setup_paths(args: argparse.Namespace) -> DataPaths:
    """Create and validate all necessary paths."""
    base_path = Path(args.uq_path)
    
    if args.dataset_name.startswith(('arctique','lidc', 'lizard')):
        main_folder_name = "UQ_maps" if not args.spatial else "UQ_spatial"
        uq_maps_path = base_path.joinpath(main_folder_name)
        
        metadata_path = base_path.joinpath("UQ_metadata") if args.metadata else None
        preds_path = base_path.joinpath("UQ_predictions")
        metrics_path = base_path.joinpath("Performance_metrics")
            
        if args.variation and args.dataset_name.startswith('arctique'):
            data_path = Path(args.label_path).joinpath(args.variation) 
        elif args.variation and args.dataset_name.startswith('lidc'):
            cycle = 'FirstCycle'
            folder = f'{args.variation}_fold0_seed123'
            placehold = 'Softmax'
            data_path = Path(args.label_path).joinpath(f'{cycle}/{placehold}/test_results/{folder}/') 
        elif args.dataset_name.startswith('lizard'): 
            data_path = Path(args.label_path)
            
    elif args.dataset_name.startswith('gta'): 
        if args.variation:
            data_path = Path(args.label_path).joinpath('CityScapesOriginalData', 'preprocessed')
        else:
            data_path = Path(args.label_path).joinpath('OriginalData', 'preprocessed')
            preds_path = base_path
            
    output_dir = Path.cwd().joinpath('output')
    output_dir.mkdir(exist_ok=True)

    for path in [uq_maps_path, data_path, preds_path]: # Validate paths - we exclude for now preds_path
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        
    metrics_path=metrics_path if metrics_path.exists() else None 
    return DataPaths(
        uq_maps=uq_maps_path,
        metadata=metadata_path,
        predictions=preds_path,
        data=data_path,
        metrics=metrics_path,
        output=output_dir
    )

# ---- Sanity Check ----

def reorder_gt_data_by_metadata(gt_list, gt_labels, metadata_zr, metadata_r, dataset): # TODO - understand whether gt_labels should also be re-indexed !
    """Reorder gt_list and gt_labels to match the order in cached_maps metadata."""
    # Extract sample names from dataset
    id_sample_names = dataset['id_masks'].dataset.sample_names
    ood_sample_names = dataset['ood_masks'].dataset.sample_names
    
    # Combine all current sample names and metadata order
    current_sample_names = list(id_sample_names) + list(ood_sample_names)
    target_order = list(metadata_zr) + list(metadata_r)
    
    # Create mapping from sample name to current index
    name_to_current_idx = {name: idx for idx, name in enumerate(current_sample_names)}
    
    # Create reordering indices based on target metadata order
    reorder_indices = []
    for target_name in target_order:
        if target_name in name_to_current_idx:
            reorder_indices.append(name_to_current_idx[target_name])
        else:
            raise ValueError(f"Sample name {target_name} not found in current data")
    return gt_list[reorder_indices], gt_labels[reorder_indices] # Reorder the arrays

def reorder_id_gt_data_by_metadata(gt_list, current_sample_names, indices): # TODO - understand whether gt_labels should also be re-indexed !
    """Reorder id gt_list to match the order in cached_maps metadata."""
    
    # Create mapping from sample name to current index
    name_to_current_idx = {name: idx for idx, name in enumerate(current_sample_names)}
    
    # Create reordering indices based on target metadata order
    reorder_indices = []
    for target_name in indices:
        if target_name in name_to_current_idx:
            reorder_indices.append(name_to_current_idx[target_name])
        else:
            raise ValueError(f"Sample name {target_name} not found in current data")
    return gt_list[reorder_indices,] # Reorder the arrays

def validate_indices(args, metadata_path, uq_method, dataset, gt_list, dataset_name):
    meta_type = f"{args.task}_noise_{args.model_noise}_{args.variation}_{args.image_noise}_{uq_method}_{args.decomp}_sample_idx.npy"
    metadata_file_path = metadata_path.joinpath(meta_type)
        
    if metadata_file_path.exists():
        indices = np.load(metadata_file_path)
        
    if dataset_name.startswith(("arctique", "lidc")):
        dataset_loader = dataset.dataset # Get the names of the samples, if the dataloader was used in the previous evaluation
        if hasattr(dataset_loader, 'sample_names') and (dataset_loader.sample_names == indices).any():
            print('✓ Uncertainty values, predictions and masks indices match')
            return gt_list
        else:
            print('⚠ WARNING: Uncertainty values, predictions and masks indices DO NOT match')
            print('Reordering ground truth masks to match the order in cached_maps metadata...')
            gt_list = reorder_id_gt_data_by_metadata(gt_list, dataset_loader.sample_names, indices)
            print('✓ Uncertainty values, predictions and masks indices match')
            return gt_list
    else:
        print('✓ Uncertainty values, predictions and masks indices match')
        return gt_list

def validate_metric_keys(metric_dict, metrics_path, task, ood_variation, data_noise, uq_method):
    """Validates that keys in metric_dict match the metadata indices stored in the corresponding .npy file."""
    # Construct path to metadata file
    base_path = Path(metrics_path).parent / "UQ_metadata"
    metadata_file = f'{task}_noise_0_{ood_variation}_{data_noise}_{uq_method}_pu_sample_idx.npy'
    metadata_path = base_path / metadata_file
    
    # Load the metadata indices
    metadata_indices = np.load(metadata_path)
    
    # Convert numpy array to list of strings (assuming indices are stored as strings in metric_dict)
    metadata_keys_list = [str(idx) for idx in metadata_indices]
    metric_keys_list = list(metric_dict.keys())
    # Check exact match (content and order)
    exact_match = metadata_keys_list == metric_keys_list
    
    # Check content match (ignoring order)
    metadata_keys_set = set(metadata_keys_list)
    metric_keys_set = set(metric_keys_list)
    content_match = metadata_keys_set == metric_keys_set
    
    # Print summary
    if exact_match:
        print(f"\n✓ Perfect Match! Metric keys and indexes of UQ maps match exactly in content and order ({len(metadata_keys_list)} items)")
    elif content_match:
        raise ValueError(f"\n⚠ CONTENT MATCH between metric keys and indexes of UQ maps, but with different order")
    else:
        raise ValueError(f"\n✗ NO MATCH between metric keys and indexes of UQ maps")

def _process_instance_predictions(pred_list: List[np.ndarray], task: str, dataset_name: str) -> List[np.ndarray]:
    """Apply instance-specific processing to predictions if needed."""
    if task == 'instance' and dataset_name.startswith(('arctique', 'lizard')):
        processed_pred_list = []
        for pred in pred_list:
            processed_inst = inst_to_3c(pred[..., 0], False)
            stacked_pred = np.stack((processed_inst, pred[..., 1]), axis=-1)
            processed_pred_list.append(stacked_pred)
        return processed_pred_list
    return pred_list

def _process_gt_masks(gt_list: List[np.ndarray], idx_task: int, dataset_name: str) -> List[np.ndarray]:
    """Apply instance-specific processing to gt masks if needed."""
    if dataset_name.startswith(('arctique', 'lizard')):
       return list(np.array(gt_list)[...,idx_task])
    return gt_list

def remove_background_only_images(gt_list, pred_list, idx_task, task, dataset_name):
    '''Excludes images containing only background (class 0) from ground truth and predictions for the AURC experiment.'''
    if dataset_name.startswith(('arctique', 'lizard')):
        mask = np.array([np.all(np.unique(gt[..., idx_task]) == 0) for gt in gt_list])
    elif dataset_name.startswith('lidc'):
        mask = np.array([np.all(np.unique(gt) == 0) for gt in gt_list])
        
    background_only_indices = np.where(mask)[0].tolist()  # Get indices of images to remove
    
    if not background_only_indices:
        pred_list = _process_instance_predictions(pred_list, task, dataset_name)
        return [], gt_list, pred_list
    
    keep_mask = ~mask  # Filter using boolean indexing
    filtered_gt_list = [gt_list[i] for i in range(len(gt_list)) if keep_mask[i]]
    filtered_pred_list = [pred_list[i] for i in range(len(pred_list)) if keep_mask[i]]
    filtered_pred_list = _process_instance_predictions(filtered_pred_list, task, dataset_name) # Apply instance processing
    
    print(f"⚠ Removed {len(background_only_indices)} images containing only background: {background_only_indices}")
    return background_only_indices, filtered_gt_list, filtered_pred_list
    
# ---- Aggregation strategies selection ----

def select_strategies(str_type: str):
    strategies = BACKGROUND_FREE_STRATEGIES if str_type == 'proportion-invariant' else AUROC_STRATEGIES
    method_names = [method for category in strategies.values() for method in category.keys()]
    # print(method_names)
    return strategies, method_names
    
# ---- Uncertainty Maps Normalization ----

def rescale_maps(unc_maps, uq_method, task, dataset_name):
    if uq_method == 'softmax':
        return unc_maps
    if task == 'instance':
        rescale_fact = np.log(3) 
    elif task == 'semantic' and dataset_name.startswith('arctique'):
        rescale_fact = np.log(6)
    elif task == 'semantic' and dataset_name.startswith('lizard'):
        rescale_fact = np.log(7) 
    elif task == 'fgbg' and dataset_name.startswith('lidc'):
        rescale_fact = np.log(2) #TODO: define how to normalize ValUES maps
    return unc_maps / rescale_fact 

# ---- Pre-cache uncertainty maps, gt and OoD AUROC targets ----

def preload_uncertainty_maps(
    uq_path: Path, 
    metadata_path: Path, 
    context_gt: List[np.ndarray], 
    gt_labels: List[np.ndarray], 
    dataset: DataLoader,
    task: str, 
    model_noise: int, 
    variation: str, 
    data_noise: str,
    dataset_name: str,
    decomp: str,
    spatial: str = None,
    ) -> Dict[str, Dict]:
    """Preload all uncertainty maps for a given noise level, their metadata indices
    and iD and OoD image targets as either 0 or 1 to then calculate the aggregators AUROC score."""
    
    uq_methods = ['softmax', 'ensemble', 'dropout', 'tta'] if decomp == 'pu' else ['ensemble', 'dropout', 'tta']
    
    # Dictionary to store loaded maps for each UQ method
    cached_maps = {}
    
    for uq_method in uq_methods:
        # Load zero-risk and noisy uncertainty maps
        uq_maps_zr, metadata_file_zr = load_unc_maps(
            uq_path, task, model_noise, variation, '0_00', uq_method, 
            decomp, dataset_name, False, metadata_path, spatial
        )
        uq_maps_r, metadata_file_r = load_unc_maps(
            uq_path, task, model_noise, variation, data_noise, uq_method, 
            decomp, dataset_name, False, metadata_path, spatial
        )
        
        # Normalize when needed
        uq_maps_zr = rescale_maps(uq_maps_zr, uq_method, task, dataset_name)
        uq_maps_r = rescale_maps(uq_maps_r, uq_method, task, dataset_name)
        
        # Concatenate maps
        uq_maps = np.concatenate((uq_maps_zr, uq_maps_r), axis=0)
        
        # Ensures that the ground truth masks, labels and uncertainty maps are consistent
        context_gt, _ = reorder_gt_data_by_metadata(
            context_gt, gt_labels, metadata_file_zr, metadata_file_r, dataset
        )
        
        # Create UncertaintyMap objects
        uncertainty_maps = [
            UncertaintyMap(array=array, mask=gt, name=None) 
            for (array, gt) in zip(uq_maps, context_gt)
        ]
        
        # Store in cache
        cached_maps[uq_method] = {
            'maps': uncertainty_maps,
            'gt_labels': gt_labels,
            'metadata': [metadata_file_zr, metadata_file_r],
        }
    return cached_maps

def create_cached_maps_from_concatenated(concatenated_data: Dict, combo_key: str) -> Dict:
    """
    Convert concatenated_data format to cached_maps format for a specific combo key.
    
    Parameters
    ----------
    concatenated_data : Dict
        Data in format: concatenated_data[uq_method][combo_key] = {'mask': ..., 'uq_map': ..., 'gt_label': ...}
    combo_key : str
        The combo key to process (e.g., '0_00_0_25', '0_00_0_50', etc.)
    
    Returns
    -------
    Dict
        cached_maps in the expected format for evaluate_all_strategies
    """
    cached_maps = {}
    
    for uq_method, combo_data in concatenated_data.items():
        if combo_key in combo_data:
            data = combo_data[combo_key]
            
            # Extract data
            masks = data['mask']  # Shape: (N, H, W) or similar
            uq_maps = data.get('uq_map', None)  # Shape: (N, H, W) or similar
            gt_labels = data['gt_label']  # Shape: (N,)
            
            # Create UncertaintyMap objects
            uncertainty_maps = []
            for i in range(len(masks)):
                uncertainty_map = UncertaintyMap(
                    array=uq_maps[i], 
                    mask=masks[i], 
                    name=None
                )
                uncertainty_maps.append(uncertainty_map)
                    
            # Store in cached_maps format
            cached_maps[uq_method] = {
                'maps': uncertainty_maps,
                'gt_labels': gt_labels,
                'metadata': None  # Add metadata if available
            }
    
    return cached_maps

def generate_combo_keys(noise_levels: List[str]) -> List[str]:
    """
    Generate combo keys from noise levels.
    Assumes combo keys are in format: base_noise_level (e.g., '0_00_0_25', '0_00_0_50')
    
    Parameters
    ----------
    noise_levels : List[str]
        List of noise levels (e.g., ['0_00', '0_25', '0_50', '0_75', '1_00'])
    
    Returns
    -------
    List[str]
        List of combo keys
    """
    combo_keys = []
    base_noise = noise_levels[0]  # Assume first noise level is the base (e.g., '0_00')
    
    for noise_level in noise_levels[1:]:  # Skip the base noise level
        combo_key = f"{base_noise}_{noise_level}"
        combo_keys.append(combo_key)
    
    return combo_keys

# ---- Aggregate uncertainty maps. ----

def process_aggr_unc(uq_path: Path, 
                     gt_sem: np.ndarray, 
                     task: str, 
                     model_noise: int, 
                     uq_method: str, 
                     decomp: str, 
                     variation: str, 
                     data_noise: str, 
                     method: callable, 
                     param: Any,
                     category: str, 
                     ind_to_rem: List,
                     dataset_name: str,
                     metadata_path: Path):
    """Aggregate uncertainty values with aggrigators' methods"""      
    # Load uncertainty maps
    uq_maps, meta = load_unc_maps(uq_path=uq_path, 
                            task=task, 
                            model_noise=model_noise, 
                            variation=variation, 
                            data_noise=data_noise, 
                            uq_method=uq_method, 
                            decomp=decomp, 
                            dataset_name=dataset_name,
                            calibr=(dataset_name== 'arctique' or dataset_name== 'lizard'),
                            metadata_path=metadata_path
                            )
    uq_maps = rescale_maps(uq_maps, uq_method, task, dataset_name)
    uq_maps = [uqmap for i, uqmap in enumerate(uq_maps) if i not in ind_to_rem]
    uq_maps = [UncertaintyMap(array=array, mask=gt, name=None) for array, gt in zip(uq_maps, gt_sem)]
    
    # Apply aggregation method to each map
    if category == 'Context-aware':
        res = [method(map, param, True) for map in uq_maps]
        # Convert numpy types to Python types for consistency - in some cases the resulting values have a weird np.float(64) format..
        converted_res = []
        for item in res:
            if hasattr(item, 'tolist'):
                converted_res.append(item.tolist())
            elif isinstance(item, (list, tuple)):
                converted_res.append([x.tolist() if hasattr(x, 'tolist') else x for x in item])
            else:
                converted_res.append(item)
        return list(zip(*converted_res))

    res = [method(map, param) for map in uq_maps]

    # Convert numpy types to regular Python types for all cases
    if hasattr(res[0], 'tolist'):
        # If elements are numpy arrays
        res = [item.tolist() if not isinstance(item, (int, np.integer)) else item for item in res]
    elif any(hasattr(item, 'item') for item in res):
        # If elements are numpy scalars
        res = [item.item() if hasattr(item, 'item') else item for item in res]
    if category == 'Threshold':
        res = np.nan_to_num(np.array(res), nan=0).tolist()
    return res, None 
        
# ---- Data Loading Functions ----

@lru_cache(maxsize=32)
def load_unc_maps(
        uq_path: Path, 
        task: str, 
        model_noise: int, 
        variation: str, 
        data_noise: str, 
        uq_method: str, 
        decomp: str,
        dataset_name: str,
        calibr: bool = False,
        metadata_path: str = None,
        spatial: str = None
    ) -> np.ndarray:
    '''Load UQ maps'''
    map_type = f"{task}_noise_{model_noise}_{variation}_{data_noise}_{uq_method}_{decomp}"
    if spatial:
        map_type += f'_{spatial}'
    
    if dataset_name.startswith(('arctique', 'lizard')):
        """Load uncertainty maps"""
        # if calibr and variation != "nuclei_intensity": map_type += "_calib"  #TODO - recheck how the predictions were calibrated and recreate them 
        map_type += ".npy"
        map_file = uq_path.joinpath(map_type)
    elif dataset_name.startswith('lidc'):
        map_type += ".npy"
        map_file = uq_path.joinpath(map_type)
        
    print(f"Loading uncertainty map: {map_type}")
    if metadata_path:
            meta_type = f"{task}_noise_{model_noise}_{variation}_{data_noise}_{uq_method}_pu_sample_idx.npy"
            meta_file = metadata_path.joinpath(meta_type)
            print(f"Loading metadata file: {meta_type}")
            return np.load(map_file), np.load(meta_file)
    return np.load(map_file) 

def load_predictions(
        paths: DataPaths,
        model_noise: int,
        variation: str,
        image_noise: str,
        uq_method: str,
        dataset_name: str,
        # calibr: bool = False
    ) -> List[np.ndarray]:
    '''Load UQ model predictions'''
    if dataset_name.startswith(('arctique', 'lizard')):
        # Load panoptic model predictions
        preds_inst_type = f"instance_noise_{model_noise}_{variation}_{image_noise}_{uq_method}"
        # if variation != "nuclei_intensity": preds_inst_type += "_calib" #TODO - recheck how the predictions were calibrated and recreate them 
        preds_inst_type += ".npy"
        preds_sem_type = f"semantic_noise_{model_noise}_{variation}_{image_noise}_{uq_method}"
        # if variation != "nuclei_intensity": preds_sem_type += "_calib"
        preds_sem_type += ".npy"
        
        preds_file_path_inst = paths.predictions.joinpath(preds_inst_type)
        preds_file_path_sem = paths.predictions.joinpath(preds_sem_type)
        
        preds_inst, preds_sem = np.load(preds_file_path_inst), np.load(preds_file_path_sem)
        return np.stack((preds_inst, preds_sem), axis=-1) #list(np.stack((preds_inst, preds_sem), axis=-1))
     
    elif dataset_name.startswith('lidc'):
        preds_type = f"fgbg_noise_{model_noise}_{variation}_{image_noise}_{uq_method}.npy"
        preds_file_path = paths.predictions.joinpath(preds_type)
        return np.load(preds_file_path) #list(np.load(preds_file_path))

def extract_gt_masks_labels(dataset_dict: Dict[str, DataLoader], task: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract ground truth masks from id and ood datasets and create binary labels for them."""
    idx_task = 2 if task == 'instance' else 1
    id_gt_list = np.array([label.numpy().squeeze() for _, label in dataset_dict['id_masks']])  # Extract id masks
    id_gt_processed = id_gt_list[..., idx_task] if id_gt_list.ndim > 3 else id_gt_list  #panoptic masks vs non-panoptic case
    
    # Check if id and ood datasets are the same (arctique/lizard case)
    if dataset_dict['id_masks'] is dataset_dict['ood_masks']:
        context_gt = np.concatenate((id_gt_processed, id_gt_processed), axis=0)
        
        # Create binary labels: first half id (0), second half ood (1)
        num_samples = len(dataset_dict['id_masks'].dataset) 
        gt_labels = np.concatenate([
            np.zeros(num_samples),
            np.ones(num_samples)
        ], axis=0)
        
    else: # Different (non-panoptic) datasets (lidc case) with different id and ood masks 
        ood_gt_list = np.array([label.numpy().squeeze() for _, label in dataset_dict['ood_masks']])
        # Concatenate different masks
        context_gt = np.concatenate((id_gt_processed, ood_gt_list), axis=0)
        
        # Create binary labels: first half id (0), second half ood (1)
        num_id_samples = len(dataset_dict['id_masks'].dataset)
        num_ood_samples = len(dataset_dict['ood_masks'].dataset)
        gt_labels = np.concatenate([
            np.zeros(num_id_samples),
            np.ones(num_ood_samples)
        ], axis=0)
    return context_gt, gt_labels

def load_dataset(
    data_path: Path,
    image_noise: str,
    num_workers: int,
    dataset_name: str,
    task: str = 'semantic',
    return_id_only: bool = False
    ) -> Tuple[Dict[str, DataLoader], np.ndarray]:
    """Load uq data loader and gt"""
    
    dataset = {'id_masks': None, 'ood_masks': None} 
    ood_values = {'id_masks': False, 'ood_masks': True}
    
    for key, val in dataset.items():
        if dataset_name.startswith("arctique"):
            # Create single data loader for arctique
            data_loader = renderHE_UQ_HVNext(
                root_dir=data_path,
                mode="test",
                image_noise=image_noise
            )
            
        elif dataset_name.startswith("lizard"):
            # Create single data loader for lizard
            data_loader = LizardDataset(
                root_dir=f"{data_path}",
                mode="test"
            )
            
        elif dataset_name.startswith("lidc"):
            # Create ID dataset (OOD=False)
            data_loader = LIDC_UQ_Dataset(
                root_dir=data_path,
                mode="test",
                OOD=ood_values[key],
                consensus_threshold=2,
                return_2d_slices=True
            )
            
        # Create DataLoader
        loader = DataLoader(
            data_loader,
            batch_size=1,
            shuffle=False,
            prefetch_factor=2,
            num_workers=num_workers,
            pin_memory=True
        )
        
        dataset[key] = loader
        if dataset_name.startswith(("arctique", "lizard")):
            dataset['ood_masks'] = dataset['id_masks']
            break
    
    # Handle early return for id-only case (for selective classification)
    if return_id_only:
        id_gt_list = np.array([label.numpy().squeeze() for _, label in dataset['id_masks']])
        print(f"GT masks shape: {id_gt_list.shape}")
        print(f"✓ Loaded {dataset_name} id-only test set and ground truth")
        return dataset['id_masks'], id_gt_list
    
    # Create ground truth labels using separated function
    context_gt, gt_labels = extract_gt_masks_labels(dataset, task)
    print(f"GT masks shape: {context_gt.shape}")
    print(f"✓ Loaded {dataset_name} test set and ground truth")
    return dataset, context_gt, gt_labels

def concatenate_dataset_results(datasets: Dict[str, Dict[str, DataLoader]], 
                               noise_combinations: List[List[str]], 
                               task: str,
                               dataset_name: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Concatenate dataset results for different noise level combinations.
    
    Args:
        datasets: Dictionary with structure {uq_method: {noise_level: DataLoader}}
        noise_combinations: List of noise level combinations to process
                           e.g., [['0_00', '0_25'], ['0_00', '0_50'], ['0_00', '0_175']]
    
    Returns:
        Dictionary with structure:
        {
            uq_method: {
                'noise_combo_key': {
                    'masks': concatenated_masks,
                    'uq_maps': concatenated_uq_maps, 
                    'gt_labels': ground_truth_labels
                }
            }
        }
    """
    concatenated_results = {}
    idx_task = 1 if task in {'instance', 'semantic'} else 2 #for panoptic masks

    for uq_method, noise_loaders in datasets.items():
        print(f"Processing UQ Method: {uq_method}")
        concatenated_results[uq_method] = {}
        
        for noise_combo in noise_combinations:
            print(f"Processing noise combo: {noise_combo}")
            # Create a key for this noise combination
            combo_key = '_'.join(sorted(noise_combo))
            
            all_masks = []
            all_uq_maps = []
            all_gt_labels = []
            
            for noise_level in noise_combo:
                if noise_level not in noise_loaders:
                    print(f"Warning: Noise level {noise_level} not found for {uq_method}")
                    continue
                
                loader = noise_loaders[noise_level]
                
                # Extract all samples from this loader
                for batch in loader:
                    # Extract masks and uq_maps from batch
                    if 'mask' in batch:
                        masks = batch['mask'].numpy() if isinstance(batch['mask'], torch.Tensor) else batch['mask']
                        masks = masks[..., idx_task] if masks.ndim > 3 else masks #panoptic masks vs non-panoptic case
                        all_masks.append(masks)
                    
                    if 'uq_map' in batch:
                        uq_maps = batch['uq_map'].numpy() if isinstance(batch['uq_map'], torch.Tensor) else batch['uq_map']
                        uq_maps = rescale_maps(uq_maps, uq_method, task, dataset_name)
                        all_uq_maps.append(uq_maps)
                    
                    # Create gt_labels: 1 for noisy data, 0 for clean (0_00)
                    batch_size = masks.shape[0] if 'mask' in batch else uq_maps.shape[0]
                    if noise_level == '0_00':
                        gt_labels = np.zeros(batch_size, dtype=int)
                    else:
                        gt_labels = np.ones(batch_size, dtype=int)
                    
                    all_gt_labels.append(gt_labels)
            
            # Concatenate all arrays for this combination
            if all_masks:
                concatenated_results[uq_method][combo_key] = {
                    'mask': np.concatenate(all_masks, axis=0),
                    'uq_map': np.concatenate(all_uq_maps, axis=0) if all_uq_maps else None,
                    'gt_label': np.concatenate(all_gt_labels, axis=0)
                }
            else:
                print(f"Warning: No data found for combination {combo_key} in {uq_method}")
    
    return concatenated_results


def create_noise_combinations(noise_levels: List[str]) -> List[List[str]]:
    """Create all possible pairwise combinations of noise levels including 0_00."""
    from itertools import combinations
    
    # Ensure 0_00 is included
    if '0_00' not in noise_levels:
        raise ValueError('iD data required for the experiment')
    
    # Create all pairwise combinations
    noise_combinations = []
    for combo in combinations(noise_levels, 2):
        # Only include combinations that have 0_00 as first position:
        if '0_00' in combo[0] and combo[0] != combo[1]:
            noise_combinations.append(list(combo))
    return noise_combinations

# Example usage:
def process_concatenated_datasets(datasets, image_noises, task, dataset_name):
    """
    Process and concatenate datasets for all noise combinations.
    """
    # Create noise combinations
    noise_combinations = create_noise_combinations(image_noises)
    
    print(f"Processing {len(noise_combinations)} noise combinations:")
    for combo in noise_combinations:
        print(f"  - {combo}")
    
    # Concatenate results
    concatenated_data = concatenate_dataset_results(datasets, noise_combinations, task, dataset_name)
    
    # Print summary
    for uq_method, combos in concatenated_data.items():
        print(f"\nUQ Method: {uq_method}")
        for combo_key, data in combos.items():
            masks_shape = data['mask'].shape if data['mask'] is not None else "None"
            uq_maps_shape = data['uq_map'].shape if data['uq_map'] is not None else "None"
            gt_labels_shape = data['gt_label'].shape
            
            print(f"  Combination {combo_key}:")
            print(f"    Masks shape: {masks_shape}")
            print(f"    UQ maps shape: {uq_maps_shape}")
            print(f"    GT labels shape: {gt_labels_shape}")
            print(f"    GT labels distribution: {np.bincount(data['gt_label'])}")
    return concatenated_data


def load_dataset_abstract_class(
    paths: DataPaths,
    image_noises: List[str],
    extra_info: dict,
    num_workers: int,
    dataset_name: str,
    task: str = 'semantic',
    return_id_only: bool = False,
    uq_methods: Optional[List[str]] = None,
) -> Tuple[Dict[str, DataLoader]]:
    """Load uq data loader and gt"""
    
    # Common setup
    noise_levels_to_process = ['0_00'] if return_id_only else image_noises
    datasets = {}
    
    # Dataset-specific configuration
    if dataset_name.startswith("arctique"):
        dataset_config = _get_arctique_config(paths, task)
    elif dataset_name.startswith("lidc"):
        dataset_config = _get_lidc_config(paths)
    elif dataset_name.startswith("gta"):
        dataset_config = _get_gta_config(paths)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented in optimized version")
    
    # Common processing loop for all datasets
    for uq_method in uq_methods:
        datasets[uq_method] = {}
        
        # Update extra_info for current UQ method
        current_extra_info = extra_info.copy()
        current_extra_info['uq_method'] = uq_method
        
        # Process each noise level for current UQ method
        for noise in noise_levels_to_process:
            current_extra_info['data_noise'] = noise
            
            # Get dataset-specific paths and configuration
            dataset_paths = dataset_config['get_paths'](noise, current_extra_info)
            
            # Create dataset using factory function
            data_loader = dataset_config['dataset_class'](
                **dataset_paths,
                uq_map_path=paths.uq_maps,
                prediction_path=paths.predictions,
                semantic_mapping_path='abc',
                **dataset_config.get('extra_kwargs', {}),
                **current_extra_info
            )
            
            # Create DataLoader with common configuration
            loader = DataLoader(
                data_loader,
                batch_size=1,
                shuffle=False,
                prefetch_factor=2,
                num_workers=num_workers,
                pin_memory=True
            )
            
            datasets[uq_method][noise] = loader
            
            # Handle early return for id-only case
            if return_id_only:
                break
    
    # Final processing
    if datasets is not None:
        concatenated_data = process_concatenated_datasets(datasets, image_noises, task, dataset_name)
        return concatenated_data
    
    # Handle early return for id-only case (for selective classification)
    # if return_id_only:
    #     id_gt_list = np.array([label.numpy().squeeze() for _, label in dataset['id_masks']])
    #     print(f"GT masks shape: {id_gt_list.shape}")
    #     print(f"✓ Loaded {dataset_name} id-only test set and ground truth")
    #     return dataset['id_masks'], id_gt_list
    
    # # Create ground truth labels using separated function
    # context_gt, gt_labels = extract_gt_masks_labels(dataset, task)
    # print(f"GT masks shape: {context_gt.shape}")
    # print(f"✓ Loaded {dataset_name} test set and ground truth")
    # return dataset, context_gt, gt_labels


def _get_arctique_config(paths: DataPaths, task: str) -> dict:
    """Get configuration for Arctique dataset"""
    # Initialize shared mask cache
    mask_cache = SharedMaskCache()
    
    # Load masks once using reference noise level
    ref_mask_path = paths.data.joinpath('0_00', 'masks')
    ref_image_path = paths.data.joinpath('0_00', 'images')
    
    sample_names = [int(digits) for filename in os.listdir(ref_image_path)
                   if (digits := ''.join(filter(str.isdigit, filename)))]
    
    # Cache masks once
    shared_masks = mask_cache.get_masks(ref_mask_path, sample_names, task)
    
    def get_paths(noise: str, extra_info: dict) -> dict:
        """Get paths for Arctique dataset - uses reference paths for all noise levels"""
        return {
            'image_path': ref_image_path,
            'mask_path': ref_mask_path,
        }
    
    return {
        'dataset_class': OptimizedArctiqueDataset,
        'get_paths': get_paths,
        'extra_kwargs': {'shared_masks': shared_masks}
    }


def _get_lidc_config(paths: DataPaths) -> dict:
    """Get configuration for LIDC dataset"""
    
    def get_paths(noise: str, extra_info: dict) -> dict:
        """Get paths for LIDC dataset - different paths based on noise level"""
        if extra_info['data_noise'] == "0_00":
            data_dir = paths.data / "id"
        else:
            data_dir = paths.data / "ood"
        
        return {
            'image_path': data_dir / "input",
            'mask_path': data_dir / "gt_seg",
        }
    
    return {
        'dataset_class': OptimizedLIDCDataset,
        'get_paths': get_paths,
        'extra_kwargs': {}
    }
    
def _get_gta_config(paths: DataPaths) -> dict:
    """Get configuration for GTA/Cityscapes dataset"""
    
    def get_paths(noise: str, extra_info: dict) -> dict:
        """Get paths for Arctique dataset - uses reference paths for all noise levels"""
        return {
            'image_path': paths.data.joinpath('images'),
            'mask_path': paths.data.joinpath('labels')
        }
    
    return {
        'dataset_class': GTA_CityscapesDataset,
        'get_paths': get_paths,
        'extra_kwargs': {}
    }