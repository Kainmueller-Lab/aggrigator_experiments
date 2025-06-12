from abc import ABC, abstractmethod
from typing import Callable, Optional


class Dataset(ABC):
    """Abstract class to define the structure of a dataset.

    Args:
        image_path (str): Path to the local directory where the images are stored.
        mask_path (str): Path to the local directory where the masks are stored.
        uq_map_path (str): Path to the local directory where the uncertainty maps are stored.
        prediction_path (str): Path to the local directory where the predictions are stored.
        semantic_mapping_path (str): Path where the semantic mapping is stored.


    """
    def __init__(self, image_path: str, mask_path: str, uq_map_path: str, prediction_path: str, semantic_mapping_path:str):
        self.image_path = image_path
        self.mask_path = mask_path
        self.uq_map_path = uq_map_path
        self.prediction_path = prediction_path
        self.semantic_mapping_path = semantic_mapping_path

    @abstractmethod
    def __len__(self):
        """Return the length / number of samples of the dataset."""
        raise NotImplementedError
    

    @abstractmethod
    def __getitem__(self, idx):
        """Return a dictionary with sample at given index. 

        Code should look like this 

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
        
        """

        raise NotImplementedError
    
    @abstractmethod
    def get_image(self, idx):
        """Return the image at the given index."""
        raise NotImplementedError

    @abstractmethod
    def get_mask(self, idx):
        """Return the mask at the given index."""
        raise NotImplementedError
    
    @abstractmethod
    def get_uq_map(self, idx):
        """Return the uq_map at the given index."""
        raise NotImplementedError
    
    @abstractmethod
    def get_prediction(self, idx):
        """Return the prediction at the given index."""
        raise NotImplementedError

    @abstractmethod
    def get_sample_name(self, idx):
        """Return the sample name at the given index."""
        raise NotImplementedError

    @abstractmethod
    def get_sample_names(self):
        """Return the list of sample names."""
        raise NotImplementedError
    
    @abstractmethod
    def get_semantic_mapping(self, idx):
        """Return the semantic mapping dictionary.
        
        Should look like this: 

        semantic_mapping = {0: 'background', 1: 'class1', 2: 'class2', ...}
        return semantic_mapping
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_info(self):
        """Return a dictionary with information about the dataset.
        
        The info_dictionary should look like this:

        info_dictionary =  {
            'image_path': self.image_path,
            'mask_path': self.mask_path,
            'uq_map_path': self.uq_map_path,
            'prediction_path': self.prediction_path,
            'semantic_mapping': self.get_semantic_mapping(),
            'datset_size': len(self),

            'task': segmentation task
            'num_classes': number of classes in the dataset,
            'semantic_mapping': self.get_semantic_mapping(),

            'uq_method': 'method used for uncertainty estimation',
            'decomposition': AU, EU or PU,

            'metadata': link to optional additional metadataset containing e.g info about the creation of uq maps (number of MC samples, augmentations for TTA...)

        }

        """

        raise NotImplementedError

