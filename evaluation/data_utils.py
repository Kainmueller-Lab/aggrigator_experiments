import numpy as np
import torch
import mahotas as mh
import os

from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from functools import lru_cache
from pathlib import Path
from PIL import Image
from typing import List, Tuple

@dataclass
class DataPaths:
    """Container for all data paths used in the program."""
    uq_maps: Path
    metadata: Path
    predictions: Path
    data: Path
    output: Path
    
# ---- Data Loading Functions ----

@lru_cache(maxsize=32)
def load_unc_maps(
        uq_path: Path, 
        task: str, 
        model_noise: int, 
        variation: str, 
        data_noise: str, 
        uq_method: str, 
        decomp: str
    ) -> np.ndarray:
    """Load uncertainty maps"""
    map_type = f"{task}_noise_{model_noise}_{variation}_{data_noise}_{uq_method}_{decomp}.npy"
    map_file = uq_path.joinpath(map_type)
    print(f"Loading uncertainty map: {map_file}")
    return np.load(map_file)

def load_predictions(
        paths: DataPaths,
        model_noise: int,
        variation: str,
        image_noise: str,
        uq_method: str
    ) -> List[np.ndarray]:
    """Load panoptic model predictions"""
    preds_inst_type = f"instance_noise_{model_noise}_{variation}_{image_noise}_{uq_method}.npy"
    preds_sem_type = f"semantic_noise_{model_noise}_{variation}_{image_noise}_{uq_method}.npy"
    
    preds_file_path_inst = paths.predictions.joinpath(preds_inst_type)
    preds_file_path_sem = paths.predictions.joinpath(preds_sem_type)
    
    preds_inst, preds_sem = np.load(preds_file_path_inst), np.load(preds_file_path_sem)
    preds = np.stack((preds_inst, preds_sem), axis=-1)    
    return list(preds)

def load_dataset(
        data_path: Path,
        image_noise: str,
        is_ood: bool,
        num_workers: int
    ) -> Tuple[DataLoader, np.ndarray]:
    """Load uq data loader and gt"""

    data_loader = renderHE_UQ_HVNext(
        data_path, 
        'test', 
        OOD=is_ood, 
        image_noise=image_noise
    )
    
    dataset = DataLoader(
        data_loader, 
        batch_size=1, 
        shuffle=False, 
        prefetch_factor=2,
        num_workers=num_workers,
        pin_memory=True
    )
    
    gt_list = np.array([label.numpy().squeeze() for _, label in dataset])
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
