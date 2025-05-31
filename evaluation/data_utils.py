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
from typing import Dict, List, Tuple, NamedTuple, Any

from aggrigator.uncertainty_maps import UncertaintyMap
from datasets.LIDC.lidc_dataset_creation import LIDC_UQ_Dataset
from datasets.Arctique.arctique_dataset_creation import renderHE_UQ_HVNext, inst_to_3c
from datasets.Lizard.lizard_dataset_creation import LizardDataset
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
    uq_maps_path = base_path.joinpath("UQ_maps")
    metadata_path = base_path.joinpath("UQ_metadata")
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
        
    output_dir = Path.cwd().joinpath('output')
    output_dir.mkdir(exist_ok=True)

    for path in [uq_maps_path, metadata_path, data_path, preds_path]: # Validate paths - we exclude for now preds_path
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

def validate_indices(args, metadata_path, uq_method, dataset, dataset_name):
    meta_type = f"{args.task}_noise_{args.model_noise}_{args.variation}_{args.image_noise}_{uq_method}_{args.decomp}_sample_idx.npy"
    metadata_file_path = metadata_path.joinpath(meta_type)
        
    if metadata_file_path.exists():
        indices = np.load(metadata_file_path)
        
    if dataset_name.startswith("arctique"):
        dataset_loader = dataset.dataset # Get the names of the samples, if the dataloader was used in the previous evaluation
        if hasattr(dataset_loader, 'sample_names') and (dataset_loader.sample_names == indices).any():
            print('✓ Uncertainty values, predictions and masks indices match')
        else:
            print('⚠ WARNING: Uncertainty values, predictions and masks indices DO NOT match')
    else:
        print('✓ Uncertainty values, predictions and masks indices match')

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
    task: str, 
    model_noise: int, 
    variation: str, 
    data_noise: str,
    dataset_name: str,
    ) -> Dict[str, Dict]:
    """Preload all uncertainty maps for a given noise level, their metadata indices
    and iD and OoD image targets as either 0 or 1 to then calculate the aggregators AUROC score."""
    
    uq_methods = ['softmax', 'ensemble', 'dropout', 'tta']
    
    # Dictionary to store loaded maps for each UQ method
    cached_maps = {}
    
    for uq_method in uq_methods:
        # Load zero-risk and noisy uncertainty maps
        uq_maps_zr, metadata_file_zr = load_unc_maps(
            uq_path, task, model_noise, variation, '0_00', 
            uq_method, 'pu', False, metadata_path
        )
        uq_maps_r, metadata_file_r = load_unc_maps(
            uq_path, task, model_noise, variation, data_noise, 
            uq_method, 'pu', False, metadata_path
        )
        
        # Normalize when needed
        uq_maps_zr = rescale_maps(uq_maps_zr, uq_method, task, dataset_name)
        uq_maps_r = rescale_maps(uq_maps_r, uq_method, task, dataset_name)
        
        # Concatenate maps
        uq_maps = np.concatenate((uq_maps_zr, uq_maps_r), axis=0)
        
        # Create UncertaintyMap objects
        uncertainty_maps = [
            UncertaintyMap(array=array, mask=gt, name=None) 
            for (array, gt) in zip(uq_maps, context_gt)
        ]
        
        # Store in cache
        cached_maps[uq_method] = {
            'maps': uncertainty_maps,
            'gt_labels': gt_labels,
            'metadata': [metadata_file_zr, metadata_file_r]
        }
    return cached_maps

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
                     dataset_name: str):
    """Aggregate uncertainty values with aggrigators' methods"""      
    # Load uncertainty maps
    uq_maps = load_unc_maps(uq_path=uq_path, 
                            task=task, 
                            model_noise=model_noise, 
                            variation=variation, 
                            data_noise=data_noise, 
                            uq_method=uq_method, 
                            decomp=decomp, 
                            dataset_name=dataset_name,
                            calibr=(dataset_name== 'arctique' or dataset_name== 'lizard')
                            )
    
    uq_maps = rescale_maps(uq_maps, uq_method, task, dataset_name)
    uq_maps = [uqmap for i, uqmap in enumerate(uq_maps) if i not in ind_to_rem]
    uq_maps = [UncertaintyMap(array=array, mask=gt, name=None) for array, gt in zip(uq_maps, gt_sem)]
    
    # Apply aggregation method to each map
    if category == 'Class-based':
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
    ) -> np.ndarray:
    '''Load UQ maps'''
    map_type = f"{task}_noise_{model_noise}_{variation}_{data_noise}_{uq_method}_{decomp}"
    
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
            meta_type = f"{task}_noise_{model_noise}_{variation}_{data_noise}_{uq_method}_{decomp}_sample_idx.npy"
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