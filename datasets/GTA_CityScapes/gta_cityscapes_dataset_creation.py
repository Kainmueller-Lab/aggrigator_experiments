import sys 
import os 
import numpy as np
import json
import torch

from PIL import Image
import matplotlib.pyplot as plt

#sys.path.append("C:/Users/cwinklm/Documents/aggrigator_experiments/datasets/")
from .datasets.dataset import Dataset_Class

semantic_mapping =  {0: 'unlabeled',
                             1: 'ego vehicle',
                             2: 'rectification border',
                             3: 'out of roi',
                             4: 'static',
                             5: 'dynamic',
                             6: 'ground',
                             7: 'road',
                             8: 'sidewalk',
                             9: 'parking',
                             10: 'rail track',
                             11: 'building',
                             12: 'wall',
                             13: 'fence',
                             14: 'guard rail',
                             15: 'bridge',
                             16: 'tunnel',
                             17: 'pole',
                             18: 'polegroup',
                             19: 'traffic light',
                             20: 'traffic sign',
                             21: 'vegetation',
                             22: 'terrain',
                             23: 'sky',
                             24: 'person',
                             25: 'rider',
                             26: 'car',
                             27: 'truck',
                             28: 'bus',
                             29: 'caravan',
                             30: 'trailer',
                             31: 'train',
                             32: 'motorcycle',
                             33: 'bicycle',
                             -1: 'license plate',
                             -2: 'gta',
                             34: 'sidewalk_2',
                             35: 'person_2',
                             36: 'car_2',
                             37: 'vegetation_2',
                             38: 'road_2'}


class cityscapes_dataset(Dataset_Class):
    """Abstract class to define the structure of a dataset.

    Args:
        image_path (str): Path to the local directory where the images are stored.
        mask_path (str): Path to the local directory where the masks are stored.
        uq_map_path (str): Path to the local directory where the uncertainty maps are stored.
        prediction_path (str): Path to the local directory where the predictions are stored.
        semantic_mapping_path (str): Path where the semantic mapping is stored.
        **kwargs: Additional keyword arguments that can be passed to specific methods.
    """
    def __init__(self, image_path: str, mask_path: str, uq_map_path: str, prediction_path: str, semantic_mapping_path:str, **kwargs):
        
        self.image_path = image_path
        self.mask_path = mask_path
        self.uq_map_path = uq_map_path
        self.prediction_path = prediction_path
        self.semantic_mapping_path = semantic_mapping_path
        self.kwargs = kwargs

        self.file_names = [f.split(".")[0] for f in os.listdir(self.uq_map_path) if f.endswith(".tif" )]
        
        #"/fast/AG_Kainmueller/data/GTA_CityScapes_UQ/Dropout/test_results/fold0_seed123/id/pred_entropy/"
        # "<basefolder>/values/<uq-method>/test_results/<ckpt>/<task>/<deomposition>/"
        str_idx = self.uq_map_path.find("GTA_CityScapes_UQ")
        uq_info = self.uq_map_path[str_idx+len("GTA_CityScapes_UQ/"):].split("/")
        self.uq_method = uq_info[0].lower().split("-")[0]
        self.model_ckpt = uq_info[2]
        self.task =  uq_info[3]
        self.decomposition =  uq_info[4]

        if self.task == "ood" and "CityScapes" not in self.image_path: 
            raise FileNotFoundError("You are currently using the GTA dataset, the OoD task requires the CityScape dataset. Please adjust paths for images and masks")

        if self.task == "id" and "CityScapes" in self.image_path: 
            raise FileNotFoundError("You are currently using the CityScapes dataset, the iD task requires the GTA dataset. Please adjust paths for images and masks")

    
    
    def __len__(self):
        """Return the length / number of samples of the dataset."""
        return len(self.file_names)
    

    def __getitem__(self, idx):
        """Return a dictionary with sample at given index. """

        sample = {
            'image': self.get_image(idx),
            'mask': self.get_mask(idx),
            'uq_map': self.get_uq_map(idx),
            'prediction': self.get_prediction(idx),
            'sample_name': self.get_sample_name(idx)
            }

        return sample
    
    def get_image(self, idx):
        """Return the image at the given index."""
        filename = self.file_names[idx] + ".npy"
        img = np.load(self.image_path + filename)
        img = img.transpose(2,0,1)
        return img

    def get_mask(self, idx):
        """Return the mask at the given index."""
        filename = self.file_names[idx] + ".npy"
        mask = np.load(self.mask_path + filename)
        return mask
    
    def get_uq_map(self, idx):
        """Return the uq_map at the given index."""
        filename = self.file_names[idx] + ".tif"
        uq_map = np.array(Image.open(self.uq_map_path + filename))
        return uq_map
    
    def get_prediction(self, idx):
        """Return the prediction at the given index."""
        filename = self.file_names[idx] + "_mean" + ".png"
        pred = np.array(Image.open(self.prediction_path + filename))
        return pred

    def get_sample_name(self, idx):
        """Return the sample name at the given index."""
        return self.file_names[idx]

    def get_sample_names(self):
        """Return the list of sample names."""
        return self.file_names
    
    def get_semantic_mapping(self):
        """Return the semantic mapping dictionary.
        
        Should look like this: 

        semantic_mapping = {0: 'background', 1: 'class1', 2: 'class2', ...}
        return semantic_mapping
        """
        
        return semantic_mapping
    
    def get_info(self):
        """Return a dictionary with information about the dataset."""

        info_dictionary =  {
            'image_path': self.image_path,
            'mask_path': self.mask_path,
            'uq_map_path': self.uq_map_path,
            'prediction_path': self.prediction_path,
            'semantic_mapping': None, #self.get_semantic_mapping(),
            'datset_size': len(self),

            'task': self.task,
            'num_classes': None,
            'semantic_mapping': None, 

            'uq_method': self.uq_method, 
            'decomposition': self.decomposition, 

            'metadata': {"model_checkpoint": self.model_ckpt}
        }