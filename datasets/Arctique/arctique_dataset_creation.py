import torch
import os 
import numpy as np
import mahotas as mh

from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

# ---- Arctique Dataset Creation Functions ----

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