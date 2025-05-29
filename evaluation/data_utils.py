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
from typing import Dict, List, Tuple, NamedTuple

from aggrigator.uncertainty_maps import UncertaintyMap
from datasets.LIDC.lidc_dataset_creation import LIDC_UQ_Dataset

# ---- Data Structures ----

@dataclass
class DataPaths:
    """Container for all data paths used in the program."""
    uq_maps: Path
    metadata: Path
    predictions: Path
    data: Path
    output: Path

class AnalysisResults(NamedTuple):
    """Container for AURC analysis results."""
    mean_aurc_val: np.ndarray
    coverages: np.ndarray
    mean_selective_risks: np.ndarray
    std_selective_risks: np.ndarray

def setup_paths(args: argparse.Namespace) -> DataPaths:
    """Create and validate all necessary paths."""
    base_path = Path(args.uq_path)
    uq_maps_path = base_path.joinpath("UQ_maps")
    metadata_path = base_path.joinpath("UQ_metadata")
    preds_path = base_path.joinpath("UQ_predictions")
    
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

    for path in [uq_maps_path, metadata_path, data_path]: # Validate paths - we exclude for now preds_path
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
    
    return DataPaths(
        uq_maps=uq_maps_path,
        metadata=metadata_path,
        predictions=preds_path,
        data=data_path,
        output=output_dir
    )
    
# ---- Uncertainty Maps Normalization ----

def rescale_maps(unc_map, uq_method, task):
    if task == 'instance':
        rescale_fact = np.log(3) 
    elif task == 'semantic':
        rescale_fact = np.log(6)
    else:
        rescale_fact = np.log(2) #TODO: define how to normalize ValUES maps
    if uq_method == 'softmax':
        return unc_map
    return unc_map / rescale_fact 

# ---- Pre-cache uncertainty maps and compute AUROC targets ----

def preload_uncertainty_maps(
    uq_path: Path, 
    metadata_path: Path, 
    context_gt: List[np.ndarray], 
    gt_labels: List[np.ndarray], 
    task: str, 
    model_noise: int, 
    variation: str, 
    data_noise: str
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
        uq_maps_zr = rescale_maps(uq_maps_zr, uq_method, task)
        uq_maps_r = rescale_maps(uq_maps_r, uq_method, task)
        
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
        calibr: bool = False,
        metadata_path: str = None,
    ) -> np.ndarray:
    """Load uncertainty maps"""
    map_type = f"{task}_noise_{model_noise}_{variation}_{data_noise}_{uq_method}_{decomp}"
    if calibr and variation != "nuclei_intensity": map_type += "_calib"
    map_type += ".npy"
    map_file = uq_path.joinpath(map_type)
    print(f"Loading uncertainty map: {map_type}")
    
    if metadata_path:
        meta_type = f"{task}_noise_{model_noise}_{variation}_{data_noise}_{uq_method}_{decomp}_sample_idx"
        meta_type += ".npy"
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
        calibr: bool = False
    ) -> List[np.ndarray]:
    """Load panoptic model predictions"""
    preds_inst_type = f"instance_noise_{model_noise}_{variation}_{image_noise}_{uq_method}"
    if calibr and variation != "nuclei_intensity": preds_inst_type += "_calib"
    preds_inst_type += ".npy"
    preds_sem_type = f"semantic_noise_{model_noise}_{variation}_{image_noise}_{uq_method}"
    if calibr and variation != "nuclei_intensity": preds_sem_type += "_calib"
    preds_sem_type += ".npy"
    
    preds_file_path_inst = paths.predictions.joinpath(preds_inst_type)
    preds_file_path_sem = paths.predictions.joinpath(preds_sem_type)
    
    preds_inst, preds_sem = np.load(preds_file_path_inst), np.load(preds_file_path_sem)
    preds = np.stack((preds_inst, preds_sem), axis=-1)    
    return list(preds)

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
    is_ood: bool,
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
                OOD=is_ood,
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
        print(f"GT masks shape: {context_gt.shape}")
        print(f"✓ Loaded {dataset_name} id-only test set and ground truth")
        return dataset['id_masks'], id_gt_list
    
    # Create ground truth labels using separated function
    context_gt, gt_labels = extract_gt_masks_labels(dataset, task)
    print(f"GT masks shape: {context_gt.shape}")
    print(f"✓ Loaded {dataset_name} test set and ground truth")
    return dataset, context_gt, gt_labels

# ---- Dataset Creation Functions ----

def inst_to_3c(gt_labels, lizard =  True):
    ''' https://github.com/digitalpathologybern/hover_next_train/blob/main/src/data_utils.py'''
    borders = mh.labeled.borders(gt_labels, Bc=np.ones((3, 3)))
    mask = gt_labels > 0
    if lizard:
        return (((borders & mask) * 1) + (mask * 1))[np.newaxis, :] 
    else:
        return (((borders & mask) * 1) + (mask * 1))
    
@staticmethod
def normalize_min_max(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if mi is None:
        mi = np.min(x)
    if ma is None:
        ma = np.max(x)
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)
    return x

class LizardDataset(Dataset):
    def __init__(self, root_dir, mode, norm=True): # mode is either train, val or test 
        self.mode = mode
        self.root_dir = root_dir
        self.norm = norm
        
        if mode == 'train':
            self.images_file = 'images_train.npy'
            self.labels_file = 'labels_train.npy'  
        elif mode == 'val':
            self.images_file = 'images_val.npy'
            self.labels_file = 'labels_val.npy' 
        elif mode == 'test': 
            self.images_file = 'images_test.npy'
            self.labels_file = 'labels_test.npy' 
        
        self.images = np.load(Path(root_dir).joinpath(self.images_file))
        self.labels = np.load(Path(root_dir).joinpath(self.labels_file))
        
        #match fibroblasts to the connective tissue 
        self.class_mapping = {0: 0, 1: 6, 2: 1, 3: 3, 4: 2, 5: 4, 6: 5} 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):        
        image_tmp = self.images[idx].astype(np.float32)
        if self.norm:
            image_tmp = normalize_min_max(image_tmp, 0, 255)
        image_tmp = torch.tensor(image_tmp, dtype=torch.float32) 
        image =  image_tmp.permute(2, 0, 1) # CxHxW
        
        if self.labels is not None:
            lab_tmp = self.labels[idx].astype(np.float32)
            labinst_tmp, labsem_tmp, lab3c_tmp = lab_tmp[...,0], lab_tmp[...,1], lab_tmp[...,2]
        
        # if self.mode == 'val' or self.mode == 'test':
        labsem_tmp = self.rearrange_class(labsem_tmp)
        label = np.stack((labinst_tmp, labsem_tmp, lab3c_tmp), axis=-1)
        label = torch.tensor(label, dtype=torch.int64)
        
        return image, label
    
    def rearrange_class(self, sem_label):
        vectorized_mapping = np.vectorize(lambda x: self.class_mapping.get(x, x))
        return vectorized_mapping(sem_label)

class renderHE_UQ_HVNext(Dataset): 
    def __init__(self, root_dir, mode, OOD = False, image_noise = "0_00"): #mode is val, test
        self.root_dir = root_dir
        self.mode = mode
        self.OOD = OOD
        self.image_noise = image_noise 

        hovernext_masks_path = root_dir 
        
        self.image_dir = Path(root_dir).joinpath(f"{image_noise}/images")
        self.inst_mask_dir = Path(hovernext_masks_path).joinpath(f"{image_noise}/masks/instance_indexing") #.joinpath(self.mode).joinpath(f"masks/instance_indexing") prior to v1-0-corrected
        self.sem_mask_dir = Path(hovernext_masks_path).joinpath(f"{image_noise}/masks/semantic_indexing") #.joinpath(self.mode).joinpath(f"masks/semantic_indexing") prior to v1-0-corrected
    
        # extract the integer indices from filenames    
        self.sample_names = [int(digits) for filename in os.listdir(self.image_dir) 
                             if (digits := ''.join(filter(str.isdigit, filename)))
                             ]

    def __len__(self):
        return len(self.sample_names)

    def get_samplename(self, idx):
        return self.sample_names[idx]

    def __getitem__(self, idx):
        fn = self.get_samplename(idx)

        image_file = self.image_dir.joinpath(f"img_{fn}.png")
        image = np.array(Image.open(image_file)).astype(np.float32)
        image = image[:, :, :3] # remove alpha channel 

        if self.OOD or self.mode == 'test': 
            inst_file = self.inst_mask_dir.joinpath(f"{fn}.tif")
            sem_file = self.sem_mask_dir.joinpath(f"{fn}.tif") 
        
        inst_label = np.array(Image.open(inst_file), dtype = int)
        sem_label = np.array(Image.open(sem_file), dtype = int)

        label_3c = inst_to_3c(inst_label, False)
        label = np.stack((inst_label, sem_label, label_3c), axis=-1)
        label = torch.tensor(label, dtype=torch.long)

        image = normalize_min_max(image, 0, 255)
        image = torch.tensor(image, dtype=torch.float32) 
        image  = image.permute(2, 0, 1) # CxHxW  
        return image, label  