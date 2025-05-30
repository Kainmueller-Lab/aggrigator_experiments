import torch
import numpy as np 
import mahotas as mh 

from torch.utils.data import Dataset
from pathlib import Path

# ---- LIDC-IDRI Dataset Creation Functions ----

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