import numpy as np
from pathlib import Path
import os
import yaml
import torch
import zarr
import json
from datasets.dataset import Dataset_Class


class wormbodies_dataset(Dataset_Class):
    """Dataset for UQ maps of BBBC010.

    Args:
        image_path (str): Path to the local directory where the images are stored.
        mask_path (str): Path to the local directory where the masks are stored.
        uq_map_path (str): Path to the local directory where the uncertainty maps are stored.
        prediction_path (str): Path to the local directory where the predictions are stored.
        semantic_mapping_path (str): Path where the semantic mapping is stored.
        **kwargs: Additional keyword arguments that can be passed to specific methods.
    """
    def __init__(self, image_path: str, mask_path: str, uq_map_path: str, prediction_path: str, semantic_mapping_path:str, **kwargs):
        self.image_path = Path(image_path)
        self.mask_path = Path(mask_path)
        self.uq_map_path = Path(uq_map_path)
        self.prediction_path = Path(prediction_path)
        self.semantic_mapping_path = semantic_mapping_path
        self.kwargs = kwargs

        self.samples = [file.split(".")[0] for file in os.listdir(self.image_path)]

        f = str(self.uq_map_path)
        f = f[(f.find("wormbodies")+len("wormbodies/")):].split("/")
        self.task = f[1]
        self.uq_method = f[2]
        self.decomposition = f[3]

    def __len__(self):
        """Return the length / number of samples of the dataset."""
        return len(self.samples)
    

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
        
        img_path = Path(self.image_path).joinpath(f"{self.get_sample_name(idx)}.zarr")
        zarr_store = zarr.open(img_path, mode='r')

        # crop the image to (512, 512) shape otherwise there are issues with the model output shape 
        image = np.array(zarr_store["volumes"]["raw_bf"][:][0,4:-4, 92:-92].copy())

        return image

    def get_mask(self, idx):
        """Return the mask at the given index."""
        mask_path = Path(self.mask_path).joinpath(f"{self.get_sample_name(idx)}.zarr")
        zarr_store = zarr.open(mask_path, mode='r')

        # crop the image to (512, 512) shape otherwise there are issues with the model output shape 
        mask = np.array(zarr_store["volumes"]["gt_fgbg"][:][0,4:-4, 92:-92].copy())
        
        return mask
    
    def get_uq_map(self, idx):
        """Return the uq_map at the given index."""
        uq_map = np.load(self.uq_map_path.joinpath(f"{self.get_sample_name(idx)}.npy"))
        return uq_map
    
    def get_prediction(self, idx):
        """Return the prediction at the given index."""
        pred = np.load(self.prediction_path.joinpath(f"{self.get_sample_name(idx)}.npy"))
        return pred[0, :, :]

    def get_sample_name(self, idx):
        """Return the sample name at the given index."""
        return self.samples[idx]

    def get_sample_names(self):
        """Return the list of sample names."""
        return self.samples
    
    def get_semantic_mapping(self):
        """Return the semantic mapping dictionary."""
        semantic_mapping = {0: 'background', 1: 'foreground'}
        return semantic_mapping
    
    def get_info(self):
        """Return a dictionary with information about the dataset."""

        info_dictionary =  {
            'image_path': str(self.image_path),
            'mask_path': str(self.mask_path),
            'uq_map_path': str(self.uq_map_path),
            'prediction_path': str(self.prediction_path),
            'datset_size': len(self),

            'task': self.task,
            'num_classes': len(self.get_semantic_mapping()),
            'semantic_mapping': self.get_semantic_mapping(),

            'uq_method': self.uq_method,
            'decomposition': self.decomposition,
        }

        return info_dictionary

