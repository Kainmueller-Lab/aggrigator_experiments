import numpy as np
import torch
import mahotas as mh
import os

from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from functools import lru_cache
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple

from aggrigator.uncertainty_maps import UncertaintyMap

@dataclass
class DataPaths:
    """Container for all data paths used in the program."""
    uq_maps: Path
    metadata: Path
    predictions: Path
    data: Path
    output: Path
    
# ---- Uncertainty Maps Normalization ----

def rescale_maps(unc_map, uq_method, task):
    rescale_fact = np.log(3) if task == 'instance' else np.log(6)
    if uq_method == 'softmax':
        return unc_map
    return unc_map / rescale_fact 

# ---- Pre-cache uncertainty maps and compute AUROC targets ----

def preload_uncertainty_maps(
    uq_path: Path, 
    metadata_path: Path, 
    gt_list: List[np.ndarray], 
    task: str, 
    model_noise: int, 
    variation: str, 
    data_noise: str
    ) -> Dict[str, Dict]:
    """Preload all uncertainty maps for a given noise level, their metadata indices
    and iD and OoD image targets as either 0 or 1 to then calculate the aggregators AUROC score."""
    
    uq_methods = ['softmax', 'ensemble', 'dropout', 'tta']
    idx_task = 2 if task == 'instance' else 1
    gt_array = np.array(gt_list)[..., idx_task]
    
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
        
        # Setup context masks
        context_gt = np.concatenate([gt_array, gt_array], axis=0)
        
        # Create UncertaintyMap objects
        uncertainty_maps = [
            UncertaintyMap(array=array, mask=gt, name=None) 
            for (array, gt) in zip(uq_maps, context_gt)
        ]
        
        # Define iD and OoD targets
        gt_labels_0 = np.zeros((len(uq_maps_zr)))
        gt_labels_1 = np.ones((len(uq_maps_r)))
        gt_labels = np.concatenate((gt_labels_0, gt_labels_1), axis=0)
        
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

def load_dataset(
        data_path: Path,
        image_noise: str,
        is_ood: bool,
        num_workers: int,
        dataset_name: str
    ) -> Tuple[DataLoader, np.ndarray]:
    """Load uq data loader and gt"""
    
    if dataset_name.startswith("arctique"):        
        data_loader = renderHE_UQ_HVNext(
            data_path,
            'test',
            OOD=is_ood,
            image_noise=image_noise
        )
    elif dataset_name.startswith("lizard"): 
        data_loader =  LizardDataset(
            root_dir=f"{data_path}", 
            mode="test")
    
    dataset = DataLoader(
        data_loader, 
        batch_size=1, 
        shuffle=False, 
        prefetch_factor=2,
        num_workers=num_workers,
        pin_memory=True
    )
    
    gt_list = np.array([label.numpy().squeeze() for _, label in dataset])
    print(f"✓ Loaded {dataset_name} test set and ground truth")
    return dataset, gt_list

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
        self.sample_names = [int(''.join(filter(str.isdigit, filename))) for filename in os.listdir(self.image_dir)] 

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