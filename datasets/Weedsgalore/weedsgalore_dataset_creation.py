import sys 
import os 
import numpy as np
import json
import torch

from PIL import Image
import matplotlib.pyplot as plt

from ..dataset import Dataset_Class



class weedsgalore_dataset(Dataset):
    """Abstract class to define the structure of a dataset.

    Args:
        image_path (str): Path to the local directory where the images are stored.
        mask_path (str): Path to the local directory where the masks are stored.
        uq_map_path (str): Path to the local directory where the uncertainty maps are stored.
        prediction_path (str): Path to the local directory where the predictions are stored.
        semantic_mapping_path (str): Path where the semantic mapping is stored.
        **kwargs: Additional keyword arguments that can be passed to specific methods.
    """
    def __init__(self, image_path: str, 
                 mask_path: str, 
                 uq_map_path: str, 
                 prediction_path: str, 
                 semantic_mapping_path:str,
                 **kwargs):

        for folder in [image_path, mask_path, mask_path, prediction_path]: 
            if not os.path.exists(folder):
                raise FileNotFoundError(f"File not found: {folder}")

        self.image_path = image_path
        self.mask_path = mask_path
        self.uq_map_path = uq_map_path
        self.prediction_path = prediction_path
        self.semantic_mapping_path = semantic_mapping_path
        self.metadata = kwargs["metadata_file"]


        # extract information about the uq maps frorm their location. 
        # uq maps are expected to be saved in the following format: 
        #  "<basefolder>/weedsgalore/<input-type>_<split>/<task>/<uq_methods>/<deomposition>/"
        str_idx = self.uq_map_path.find("weedsgalore")
        uq_info = self.uq_map_path[str_idx+len("weedsgalore/"):].split("/")
        self.input_type, self.split = uq_info[0].split("_")
        self.in_bands = 3 if self.input_type == "rgb" else 5
        self.task = uq_info[1]
        self.num_classes = 6 if self.task == "semantic" else 3
        self.uq_method = uq_info[2]
        self.decomposition = uq_info[3]

        with open(image_path + f'/splits/{self.split}.txt', 'r') as file:
            data = [line.rstrip('\n') for line in file]  # Assuming elements are numeric
        self.img_list = np.array(data)

        
    def __len__(self):
        """Return the length / number of samples of the dataset."""
        return len(self.img_list)
    

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
        
        img_path = os.path.join(self.image_path, self.img_list[idx][:10], 'images', self.img_list[idx])
        red_band = plt.imread(img_path + '_R.png')
        green_band = plt.imread(img_path + '_G.png')
        blue_band = plt.imread(img_path + '_B.png')
        nir_band = plt.imread(img_path + '_NIR.png')
        re_band = plt.imread(img_path + '_RE.png')

        if self.in_bands == 3:
            img = np.stack((red_band, green_band, blue_band))
        elif self.in_bands == 5:
            img = np.stack((red_band, green_band, blue_band, nir_band, re_band))
        return img


    def get_mask(self, idx):
        """Return the mask at the given index."""
        
        # load semantic label
        label_path = os.path.join(self.mask_path, self.img_list[idx][:10], 'semantics', self.img_list[idx])
        label = Image.open(label_path + '.png')
        label = np.array(label)

        if self.task == "crops_vs_weed": 
            label[label>1] = 2
        
        return label
    

    def get_uq_map(self, idx):
        """Return the uq_map at the given index."""
        
        uq_map = np.load(self.uq_map_path+ self.get_sample_name(idx) + ".npy")
        return uq_map
    

    def get_prediction(self, idx):
        """Return the prediction at the given index."""
        
        pred = np.load(self.prediction_path+ self.get_sample_name(idx) + ".npy")       
        return pred


    def get_sample_name(self, idx):
        """Return the sample name at the given index."""
        
        return self.img_list[idx]


    def get_sample_names(self):
        """Return the list of sample names."""
        
        return self.img_list
    

    def get_semantic_mapping(self):
        """Return the semantic mapping dictionary."""
        
        if self.task == "crops_vs_weed": 
            semantic_mapping = {0:"bg", 1:"crop", 2:"weed"}
        else: 
            semantic_mapping = {0:"bg", 1:"maize", 2:"amaranth", 3:"barnyard grass", 4:"quickweed", 5:"weed other"}
        return semantic_mapping
    

    def get_info(self):
        """Return a dictionary with information about the dataset.
        """

        info_dictionary =  {
            'image_path': self.image_path,
            'mask_path': self.mask_path,
            'uq_map_path': self.uq_map_path,
            'prediction_path': self.prediction_path,
            'semantic_mapping': self.get_semantic_mapping(),
            'datset_size': len(self),

            'task': self.task,
            'num_classes': self.num_classes,

            'uq_method': self.uq_method,
            'decomposition': self.decomposition,

            'metadata': self.metadata,
            'input_typs': self.input_type, 
            'split': self.split 
        }
        return info_dictionary
