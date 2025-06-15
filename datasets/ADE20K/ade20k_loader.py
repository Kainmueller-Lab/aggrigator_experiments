import os
import numpy as np
import json

from PIL import Image

from ..dataset import Dataset_Class


class ADE20K(Dataset_Class):
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

        # Load filenames
        self.image_filenames = sorted([f.split(".")[0] for f in os.listdir(self.image_path) if f.endswith(".jpg")])
        self.check_matchig_filecount()


    def __len__(self):
        """Return the length / number of samples of the dataset."""
        return len(self.image_filenames)
    

    def __getitem__(self, idx):
        """Return a dictionary with sample at given index. """
        image = self.get_image(idx)
        mask = self.get_mask(idx)
        uq_map = self.get_uq_map(idx)
        sample_name = self.get_sample_name(idx)
        prediction = self.get_prediction(idx)

        sample = {
            'image': image,
            'mask': mask,
            'uq_map': uq_map,
            'prediction': prediction,
            'sample_name': sample_name
            }

        return sample

    
    def get_image(self, idx):
        """Return the image at the given index."""
        try:
            img_array = np.array(Image.open(os.path.join(self.image_path, self.image_filenames[idx] + '.jpg')))
            return img_array.transpose(2, 0, 1)

        except:
            print(f"Warning: Could not load image {self.image_filenames[idx]} from {self.image_path}")
            return None


    def get_mask(self, idx):
        """Return the mask at the given index."""
        try:
            return np.array(Image.open(os.path.join(self.mask_path, self.image_filenames[idx] + '.png')))
        except:
            print(f"Warning: Could not load mask {self.image_filenames[idx]} from {self.mask_path}")
            return None
    
    def get_uq_map(self, idx):
        """Return the uq_map at the given index."""
        try:
            return np.load(os.path.join(self.uq_map_path, self.image_filenames[idx] + '.npy'))
        except:
            print(f"Warning: Could not load uq_map {self.image_filenames[idx]} from {self.uq_map_path}")
    
    def get_prediction(self, idx):
        """Return the prediction at the given index."""
        try:
            return np.load(os.path.join(self.prediction_path, self.image_filenames[idx] + '.npy'))
        except:
            print(f"Warning: Could not load prediction {self.image_filenames[idx]} from {self.prediction_path}")
            return None

    def get_sample_name(self, idx):
        """Return the sample name at the given index."""
        return self.image_filenames[idx]

    def get_sample_names(self):
        """Return the list of sample names."""
        return self.image_filenames
    
    def get_semantic_mapping(self):
        """Return the semantic mapping dictionary."""
        # Load label index mapping from json file
        with open(self.semantic_mapping_path, 'r') as f:
            index_mapping = json.load(f)
        
        semantic_mapping = {idx: label_info["Name"] for idx, label_info in index_mapping.items()}
        return semantic_mapping


    def normalized_dataset_path(self, path, dataset_name):
        """
        Extracts the sub-path starting from the given dataset name and returns it with
        all path separators replaced by underscores.

        Args:
            path (str): Full directory or file path.
            dataset_name (str): Name of the dataset folder to start extraction from.

        Returns:
            str: Underscore-joined subpath starting at dataset_name (e.g., 'ADE20K_split_model').

        Example:
            >>> normalized_dataset_path('/data/UQ_maps/ADE20K/validation/deeplabv3', 'ADE20K')
            'ADE20K_validation_deeplabv3'
        """
        norm_path = os.path.normpath(path)
        parts = norm_path.split(os.sep)
        if dataset_name in parts:
            idx = parts.index(dataset_name)
            sub_parts = parts[idx:]
            return "_".join(sub_parts)
        print(f"Warning: Dataset name '{dataset_name}' not found in path: {path}")
        return ""

    
    def get_info(self):
        """Return a dictionary with information about the dataset."""
        info_dictionary =  {
            'image_path': self.image_path,
            'mask_path': self.mask_path,
            'uq_map_path': self.uq_map_path,
            'prediction_path': self.prediction_path,
            'datset_size': len(self),

            'task': 'semantic',
            'num_classes': 150,
            'semantic_mapping': self.get_semantic_mapping(),

            'uq_method': 'dropout',
            'decomposition': 'pu',
            'normalized_uq_maps': True,

            'metadata': '/fast/AG_Kainmueller/data/ADEChallengeData2016/predictions/deeplabv3_r50-d8_4xb4-160k_ade20k-512x512/metadata/',
            'dataset_name': self.normalized_dataset_path(self.uq_map_path, 'ADE20K'),
        }
        return info_dictionary
    
    
    def check_matchig_filecount(self):
        img_count = len([f for f in os.listdir(self.image_path) if f.endswith(".jpg")])
        mask_count = len([f for f in os.listdir(self.mask_path) if f.endswith(".png")])
        pred_count = len([f for f in os.listdir(self.prediction_path) if f.endswith(".npy")])
        uq_map_count = len([f for f in os.listdir(self.uq_map_path) if f.endswith(".npy")])
        
        if img_count != mask_count:
            print(f"Warning: Number of images ({img_count}) does not match number of masks ({mask_count}).")
        if img_count != pred_count:
            print(f"Warning: Number of images ({img_count}) does not match number of predictions ({pred_count}).")
        if img_count != uq_map_count:
            print(f"Warning: Number of images ({img_count}) does not match number of uncertainty maps ({uq_map_count}).")
