import os
import json
import random
import mahotas as mh
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader 

from datasets.dataset import Dataset_Class

def inst_to_3c(gt_labels, lizard =  True):
    ''' https://github.com/digitalpathologybern/hover_next_train/blob/main/src/data_utils.py'''
    borders = mh.labeled.borders(gt_labels, Bc=np.ones((3, 3)))
    mask = gt_labels > 0
    if lizard:
        return (((borders & mask) * 1) + (mask * 1))[np.newaxis, :] 
    else:
        return (((borders & mask) * 1) + (mask * 1))

# ADE20K semantic class mapping with class names and RGB colors
# Based on the official ADE20K dataset with 150 classes
ade20k_semantic_mapping = {
    0: ['wall', [120, 120, 120]],
    1: ['building', [180, 120, 120]],
    2: ['sky', [6, 230, 230]],
    3: ['floor', [80, 50, 50]],
    4: ['tree', [4, 200, 3]],
    5: ['ceiling', [120, 120, 80]],
    6: ['road', [140, 140, 140]],
    7: ['bed', [204, 5, 255]],
    8: ['windowpane', [230, 230, 230]],
    9: ['grass', [4, 250, 7]],
    10: ['cabinet', [224, 5, 255]],
    11: ['sidewalk', [235, 255, 7]],
    12: ['person', [150, 5, 61]],
    13: ['earth', [120, 120, 70]],
    14: ['door', [8, 255, 51]],
    15: ['table', [255, 6, 82]],
    16: ['mountain', [143, 255, 140]],
    17: ['plant', [204, 255, 4]],
    18: ['curtain', [255, 51, 7]],
    19: ['chair', [204, 70, 3]],
    20: ['car', [0, 102, 200]],
    21: ['water', [61, 230, 250]],
    22: ['painting', [255, 6, 51]],
    23: ['sofa', [11, 102, 255]],
    24: ['shelf', [255, 7, 71]],
    25: ['house', [255, 9, 224]],
    26: ['sea', [9, 7, 230]],
    27: ['mirror', [220, 220, 220]],
    28: ['rug', [255, 9, 92]],
    29: ['field', [112, 9, 255]],
    30: ['armchair', [8, 255, 214]],
    31: ['seat', [7, 255, 224]],
    32: ['fence', [255, 184, 6]],
    33: ['desk', [10, 255, 71]],
    34: ['rock', [255, 41, 10]],
    35: ['wardrobe', [7, 255, 255]],
    36: ['lamp', [224, 255, 8]],
    37: ['bathtub', [102, 8, 255]],
    38: ['railing', [255, 61, 6]],
    39: ['cushion', [255, 194, 7]],
    40: ['base', [255, 122, 8]],
    41: ['box', [0, 255, 20]],
    42: ['column', [255, 8, 41]],
    43: ['signboard', [255, 5, 153]],
    44: ['chest of drawers', [6, 51, 255]],
    45: ['counter', [235, 12, 255]],
    46: ['sand', [160, 150, 20]],
    47: ['sink', [0, 163, 255]],
    48: ['skyscraper', [140, 140, 140]],
    49: ['fireplace', [250, 10, 15]],
    50: ['refrigerator', [20, 255, 0]],
    51: ['grandstand', [31, 255, 0]],
    52: ['path', [255, 31, 0]],
    53: ['stairs', [255, 224, 0]],
    54: ['runway', [153, 255, 0]],
    55: ['case', [0, 0, 255]],
    56: ['pool table', [255, 71, 0]],
    57: ['pillow', [0, 235, 255]],
    58: ['screen door', [0, 173, 255]],
    59: ['stairway', [31, 0, 255]],
    60: ['river', [11, 200, 200]],
    61: ['bridge', [255, 82, 0]],
    62: ['bookcase', [0, 255, 245]],
    63: ['blind', [0, 61, 255]],
    64: ['coffee table', [0, 255, 112]],
    65: ['toilet', [0, 255, 133]],
    66: ['flower', [255, 0, 0]],
    67: ['book', [255, 163, 0]],
    68: ['hill', [255, 102, 0]],
    69: ['bench', [194, 255, 0]],
    70: ['countertop', [0, 143, 255]],
    71: ['stove', [51, 255, 0]],
    72: ['palm', [0, 82, 255]],
    73: ['kitchen island', [0, 255, 41]],
    74: ['computer', [0, 255, 173]],
    75: ['swivel chair', [10, 0, 255]],
    76: ['boat', [173, 255, 0]],
    77: ['bar', [0, 255, 153]],
    78: ['arcade machine', [255, 92, 0]],
    79: ['hovel', [255, 0, 255]],
    80: ['bus', [255, 0, 245]],
    81: ['towel', [255, 0, 102]],
    82: ['light', [255, 173, 0]],
    83: ['truck', [255, 0, 20]],
    84: ['tower', [255, 184, 184]],
    85: ['chandelier', [0, 31, 255]],
    86: ['awning', [0, 255, 61]],
    87: ['streetlight', [0, 71, 255]],
    88: ['booth', [255, 0, 204]],
    89: ['television receiver', [0, 255, 194]],
    90: ['airplane', [0, 255, 82]],
    91: ['dirt track', [0, 10, 255]],
    92: ['apparel', [0, 112, 255]],
    93: ['pole', [51, 0, 255]],
    94: ['land', [0, 194, 255]],
    95: ['bannister', [0, 122, 255]],
    96: ['escalator', [0, 255, 163]],
    97: ['ottoman', [255, 153, 0]],
    98: ['bottle', [0, 255, 10]],
    99: ['buffet', [0, 255, 133]],
    100: ['poster', [255, 0, 235]],
    101: ['stage', [171, 0, 255]],
    102: ['van', [0, 255, 0]],
    103: ['ship', [255, 0, 163]],
    104: ['fountain', [255, 204, 0]],
    105: ['conveyer belt', [122, 0, 255]],
    106: ['canopy', [0, 255, 92]],
    107: ['washer', [0, 255, 255]],
    108: ['plaything', [255, 0, 82]],
    109: ['swimming pool', [0, 255, 235]],
    110: ['stool', [0, 61, 255]],
    111: ['barrel', [0, 255, 71]],
    112: ['basket', [255, 0, 173]],
    113: ['waterfall', [0, 204, 255]],
    114: ['tent', [194, 0, 255]],
    115: ['bag', [0, 255, 184]],
    116: ['minibike', [0, 92, 255]],
    117: ['cradle', [255, 0, 224]],
    118: ['oven', [255, 0, 153]],
    119: ['ball', [0, 255, 163]],
    120: ['food', [255, 235, 0]],
    121: ['step', [0, 255, 245]],
    122: ['tank', [0, 173, 255]],
    123: ['trade name', [255, 0, 245]],
    124: ['microwave', [255, 0, 122]],
    125: ['pot', [255, 245, 0]],
    126: ['animal', [10, 190, 212]],
    127: ['bicycle', [214, 255, 0]],
    128: ['lake', [0, 204, 255]],
    129: ['dishwasher', [20, 0, 255]],
    130: ['screen', [255, 255, 0]],
    131: ['blanket', [0, 153, 255]],
    132: ['sculpture', [0, 41, 255]],
    133: ['hood', [0, 255, 204]],
    134: ['sconce', [41, 0, 255]],
    135: ['vase', [41, 255, 0]],
    136: ['traffic light', [173, 0, 255]],
    137: ['tray', [0, 245, 255]],
    138: ['ashcan', [71, 0, 255]],
    139: ['fan', [122, 0, 255]],
    140: ['pier', [0, 255, 184]],
    141: ['crt screen', [0, 92, 255]],
    142: ['plate', [184, 255, 0]],
    143: ['monitor', [0, 133, 255]],
    144: ['bulletin board', [255, 214, 0]],
    145: ['shower', [25, 194, 194]],
    146: ['radiator', [102, 255, 0]],
    147: ['glass', [92, 0, 255]],
    148: ['clock', [0, 255, 255]],
    149: ['flag', [255, 0, 245]]
}

class ADE20K_CityscapesDataset(Dataset_Class):
    """Class to define the structure of ADE20k vs CityScapes dataset.

    Args:
        image_path (str): Path to the local directory where the images are stored.
        mask_path (str): Path to the local directory where the masks are stored.
        uq_map_path (str): Path to the local directory where the uncertainty maps are stored.
        prediction_path (str): Path to the local directory where the predictions are stored.
        semantic_mapping_path (str): Path where the semantic mapping is stored.
        **kwargs: Additional keyword arguments that can be passed to specific methods.
    """
    def __init__(self, image_path: str, mask_path: str, uq_map_path: str, prediction_path: str, semantic_mapping_path:str, **kwargs):
        ###############
        # Directories examples:
        #
        # ADE20k
        # /fast/AG_Kainmueller/data/ADEChallengeData2016/images/validation/
        # /fast/AG_Kainmueller/data/ADEChallengeData2016/annotations/validation/
        # /fast/AG_Kainmueller/data/UQ_maps/ADE20K/validation_deeplabv3/semantic/dropout/pu/
        # /fast/AG_Kainmueller/data/ADEChallengeData2016/predictions/deeplabv3_r50-d8_4xb4-160k_ade20k-512x512/predictions/
        #
        # CityScapes
        # /fast/AG_Kainmueller/data/ADEChallengeData2016/images/test_cityscapes/
        # /fast/AG_Kainmueller/data/ADEChallengeData2016/annotations/test_cityscapes/
        # /fast/AG_Kainmueller/data/UQ_maps/ADE20K/ood_deeplabv3/semantic/dropout/pu/
        # /fast/AG_Kainmueller/data/GTA_copy/predictions/deeplabv3_r50-d8_4xb4-160k_ade20k-512x512/predictions/
        ################
        
        self.image_path =  Path(image_path)
        self.mask_path = Path(mask_path)
        self.uq_map_path =  Path(uq_map_path)
        self.semantic_mapping_path = semantic_mapping_path
        self.prediction_path = Path(prediction_path).parent
        
        # Extract kwargs with defaults if not provided
        self.task = kwargs.get('task', None)
        self.uq_method = kwargs.get('uq_method', None)
        self.model_noise = kwargs.get('model_noise', None)
        self.decomp = kwargs.get('decomp', None)
        self.spatial = kwargs.get('spatial', None)
        self.variation = kwargs.get('variation', None)
        self.data_noise = kwargs.get('data_noise', None)
        self.metadata = kwargs.get('metadata', False)
        self.model_checkpoint = kwargs.get('metadata', False)
        self.model_ckpt = kwargs.get('model_checkpoint', None)
        self.split = kwargs.get('split', ['test'])
        self.split_path = kwargs.get('split_path', None)
        
        # Validate required parameters
        self.__validate_required_params__()
        
        # Define the pre-trained model selected in the opnemmlab zoo during evaluation
        self.model_name = self.model_ckpt.split('_')[0]
        
        # Set up dataset-specific paths and configurations
        self.__setup_dataset_paths__()
        
        # Define boolean for element retrieval
        self.is_cityscapes = "test_cityscapes" in str(self.image_path)
        
        # Extract the integer indices from filenames
        if self.split_path:
            self.get_sample_names_from_split_file()
        else:
            self.get_sample_names_from_img_directory()

        # Load filenames
        self.check_matchig_filecount()
    
    def __setup_dataset_paths__(self):
        """Set up dataset-specific paths and validate dataset consistency."""
        # Set variation based on data_noise
        if self.data_noise in ["0_00", "1_00"] and not self.variation:
            self.variation = 'cityscapes'
        
        # Uq_map_path and prediction_path parameters must end with the checkpoint folder when passed to the class 
        if not self.uq_map_path.name.startswith('ADE20K'):
            raise ValueError(f"Invalid directory structure. Expected folder starting with 'ADE20K', got: {self.uq_map_path.name}")
        
        # Set task-specific paths
        if self.data_noise == "0_00":
            self.uq_map_path = self.uq_map_path / f'validation_{self.model_name}'
            self.prediction_path = self.prediction_path / "ADEChallengeData2016" / "predictions"
        else:
            self.uq_map_path = self.uq_map_path / f'ood_{self.model_name}' 
            self.prediction_path = self.prediction_path / "GTA_copy" / "predictions"
        
        # Validate dataset consistency
        self.__validate_dataset_consistency__()
        
        # Complete uq_map directory with either decomposition type: aleatoric, epistemic or pred_entr
        self.uq_map_path = self.uq_map_path.joinpath('semantic', self.uq_method, self.decomp)
        
        # Set prediction path
        self.prediction_path = self.prediction_path.joinpath(self.model_ckpt, self.uq_method, "predictions")
        
        # Final validation that paths exist
        if not self.uq_map_path.exists():
            raise FileNotFoundError(f"Uncertainty map path does not exist: {self.uq_map_path}")
    
    def __validate_required_params__(self):
        """Validate that required parameters are provided."""
        if not self.model_ckpt:
            raise ValueError("model_checkpoint is required in kwargs")
        
        if not self.uq_method:
            raise ValueError("uq_method is required in kwargs")
        
        if not self.data_noise:
            raise ValueError("data_noise is required in kwargs")
        
        if not self.decomp:
            raise ValueError("decomp is required in kwargs")
        
        # Validate data_noise values
        valid_data_noise = ["0_00", "1_00"]
        if self.data_noise not in valid_data_noise:
            raise ValueError(f"data_noise must be one of {valid_data_noise}, got: {self.data_noise}")
    
    def __validate_dataset_consistency__(self):
        """Validate that the correct dataset is being used for the task."""
        # Define boolean to then validate dataset consistencies
        self.is_cityscapes = "test_cityscapes" in str(self.image_path)
        
        if self.data_noise == "1_00":  # OoD task
            if not self.is_cityscapes:
                raise FileNotFoundError(
                    "OoD task (data_noise='1_00') requires CityScapes dataset. "
                    f"Current image path: {self.image_path}"
                )
        
        elif self.data_noise == "0_00":  # iD task
            if self.is_cityscapes:
                raise FileNotFoundError(
                    "iD task (data_noise='0_00') requires ADE20K dataset. "
                    f"Current image path: {self.image_path}"
                )

    def __len__(self):
        """Return the length / number of samples of the dataset."""
        return len(self.image_filenames)

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
        if self.is_cityscapes:
            filename = self.image_filenames[idx] + ".npy"
            img_array = np.load(self.image_path.joinpath(filename))
        else:
            filename = self.image_filenames[idx] + '.jpg'
            img_array = np.array(Image.open(self.image_path.joinpath(filename)))
        return img_array.transpose(2, 0, 1)

    def get_mask(self, idx):
        """Return the mask at the given index."""
        if self.is_cityscapes:  
            filename = self.image_filenames[idx] + ".npy"
            mask = np.load(self.mask_path.joinpath(filename))
        else:
            filename = self.image_filenames[idx] + '.png'
            semantic_img = Image.open(os.path.join(self.mask_path.joinpath(filename)))
            mask = np.array(semantic_img)
            if self.task.startswith('panoptic'):
                base_path = self.mask_path.parents[1]
                instance_img = Image.open(base_path.joinpath('annotations_instance', self.mask_path.name, filename))
                # Check if the dimensions of the two masks are different
                if instance_img.size != semantic_img.size:
                    # Resize the instance mask to match the semantic mask's dimensions via Image.Resampling.NEAREST 
                    mask_inst = np.array(instance_img.resize(semantic_img.size, Image.Resampling.NEAREST))
                return np.stack((mask_inst, mask), axis=-1)
        return mask
    
    def get_uq_map(self, idx):
        """Return the uq_map at the given index."""
        filename = self.image_filenames[idx] + '.npy'
        return np.load(os.path.join(self.uq_map_path.joinpath(filename)))
    
    def get_prediction(self, idx):
        """Return the prediction at the given index."""
        filename = self.image_filenames[idx] + '.npy'
        return np.load(os.path.join(self.prediction_path.joinpath(filename)))

    def get_sample_name(self, idx):
        """Return the sample name at the given index."""
        return self.image_filenames[idx]

    def get_sample_names(self):
        """Return the list of sample names."""
        return self.image_filenames
    
    def get_sample_names_from_split_file(self):
        """Load sample names from split file."""
        split_path = Path(self.split_path)
        if split_path.suffix == '.json':
            print(f"Loading sample names from JSON split file: {split_path}")
            with open(split_path, "r") as f:
                names_from_json = json.load(f)  # Assuming the JSON file contains a flat list of filenames
                self.image_filenames = [str(name).split(".")[0] for name in names_from_json] # Ensure we strip any potential file extensions
        
        else:
            with open(split_path, "r") as f:
                self.image_filenames = [
                    line.strip().split(".")[0] 
                    for line in f 
                    if line.strip().endswith(".png")
                ]
    
    def get_sample_names_from_img_directory(self):
        """Load sample names from directory listing."""
        self.image_filenames = [
            f.split(".")[0] 
            for f in os.listdir(self.image_path) 
            if (f.endswith(".npy") if self.is_cityscapes 
                else f.endswith(".jpg"))
        ]
            
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
            'num_classes': 150,
            'semantic_mapping': self.get_semantic_mapping(),
            'dataset_size': len(self),
            'task': self.task,
            'uq_method': self.uq_method, 
            'decomposition': self.decomp, 
            'metadata': self.metadata, 
            "model_checkpoint": self.model_ckpt,
            "metadata_path": '/fast/AG_Kainmueller/data/ADEChallengeData2016/predictions/deeplabv3_r50-d8_4xb4-160k_ade20k-512x512/metadata/',
            'dataset_name': self.normalized_dataset_path(self.uq_map_path, 'ADE20K'),
        }
        return info_dictionary
    
    
    def check_matchig_filecount(self):
        img_count = len([f for f in os.listdir(self.image_path) if (f.endswith(".jpg") if not self.is_cityscapes else f.endswith(".npy"))])
        mask_count = len([f for f in os.listdir(self.mask_path) if (f.endswith(".png") if not self.is_cityscapes else f.endswith(".npy"))])
        pred_count = len([f for f in os.listdir(self.prediction_path) if f.endswith(".npy")])
        uq_map_count = len([f for f in os.listdir(self.uq_map_path) if (f.endswith(".npy") and not f.startswith("cityscapes"))])
        
        if img_count != mask_count:
            print(f"Warning: Number of images ({img_count}) does not match number of masks ({mask_count}).")
        if img_count != pred_count:
            print(f"Warning: Number of images ({img_count}) does not match number of predictions ({pred_count}).")
        if img_count != uq_map_count:
            print(f"Warning: Number of images ({img_count}) does not match number of uncertainty maps ({uq_map_count}).")
            
class OptimizedADE20K_CityscapesDataset(ADE20K_CityscapesDataset):
    """Memory-efficient version that can skip loading images"""
    
    def __init__(self, image_path, mask_path, uq_map_path, prediction_path, 
                 semantic_mapping_path, load_images=False, load_preds=True, 
                 max_samples=None, random_sampling=True, seed=42, **kwargs): #max_samples=28
        super().__init__(image_path, mask_path, uq_map_path, prediction_path, 
                        semantic_mapping_path, **kwargs)
        self.load_images = load_images
        self.load_preds = load_preds
        self.random_sampling = random_sampling
        self.seed = seed
                
        # Limit the number of samples if specified
        if max_samples is not None and max_samples < len(self.image_filenames): #and not self.is_cityscapes:
            if random_sampling:
                self._random_sample_selection(max_samples)
            else:
                self.image_filenames = self.image_filenames[:max_samples]
    
    def _random_sample_selection(self, max_samples):
        """Randomly select max_samples from sample_names with reproducible seed."""
        # Set seeds for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)

        original_count = len(self.image_filenames)
        print("original_count", original_count)
        
        # Randomly sample without replacement
        self.image_filenames = random.sample(self.image_filenames, max_samples)
        
        print(f"Randomly selected {max_samples} samples from {original_count} total samples (seed={self.seed})")
    
    def resample(self, max_samples=500, new_seed=None):
        """Re-sample the dataset with a different number of samples or seed."""
        if new_seed is not None:
            self.seed = new_seed
        
        # Get all available samples again (need to reload from directory)
        if hasattr(self, 'split_path') and self.split_path:
            self.get_sample_names_from_split_file()
        else:
            self.get_sample_names_from_img_directory()
        
        # Apply random sampling with new parameters
        if max_samples < len(self.image_filenames):
            self._random_sample_selection(max_samples)
        return self
    
    def get_sampling_info(self):
        """Return information about the current sampling configuration."""
        return {
            'total_samples': len(self.image_filenames),
            'random_sampling': self.random_sampling,
            'seed': self.seed if self.random_sampling else None,
            'sample_names_preview': self.image_filenames[:5] if len(self.image_filenames) > 5 else self.image_filenames
        }
        
    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError("Index out of bounds.")
                        
        data = {
            'mask': self.get_mask(idx), 
            'uq_map': self.get_uq_map(idx),
            'sample_name': self.get_sample_name(idx),
        }
    
        # Only load images if requested
        if self.load_images:
            data['image'] = self.get_image(idx)
        
        # Only load predictions if requested
        if self.load_preds:
            data['prediction'] = self.get_prediction(idx)
        return data
            
            
# ---- Main Function to to test GTA_CityscapesDataset ----   
    
def main():
    base_path = "/fast/AG_Kainmueller/data/ADEChallengeData2016/"
    text_path = f"{os.path.dirname(base_path.rstrip('/'))}/GTA_ValUES_splits/ADE20k_id_test"
    
    extra_info = {
        'task' : 'semantic', #'panoptic'
        'variation' : 'cityscapes',
        'model_noise' : 0,
        'data_noise': '0_00',
        'uq_method': 'softmax',
        'decomp' : 'pu',
        'spatial' : None,
        'split_path' : None, #text_path,
        'split' : None,
        'metadata' : False,
        'model_checkpoint': 'deeplabv3_r50-d8_4xb4-160k_ade20k-512x512',
    }
    
    if extra_info['data_noise'] == '0_00':
        data_fold = 'validation'
    else:
        data_fold = 'test_cityscapes'
    
    image_path = f"{base_path}/images/{data_fold}"
    mask_path = f"{base_path}/annotations/{data_fold}"
    uq_map_path = f"{os.path.dirname(base_path.rstrip('/'))}/UQ_maps/ADE20K/"
    prediction_path = base_path
        
    data_loader = ADE20K_CityscapesDataset(image_path, 
                                            mask_path, 
                                            uq_map_path, 
                                            prediction_path, 
                                            '/fast/AG_Kainmueller/data/ADEChallengeData2016/objectInfo150.json',
                                            **extra_info)
        
    loader = DataLoader(data_loader, 
                        batch_size=1, 
                        shuffle=False,
                        prefetch_factor=2,
                        num_workers=4,
                        pin_memory=True
                        )
    
    sem_maps_colors = ade20k_semantic_mapping
    
    data = next(iter(loader))
    print(data['image'].shape,
          data['mask'].shape, 
          data['uq_map'].shape, 
          data['prediction'].shape, 
          data['sample_name'])
    
    def label_to_rgb(label_map, label_colors, mode='prediction'):
        """Converts a (H, W) label map to an (H, W, 3) RGB overlay."""
        h, w = label_map.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for label, color in label_colors.items():
            mask = (label_map == label)
            if mode=='instance':
                rgb[mask] = color
            else:
                rgb[mask] = color[1]
        return rgb
    
    # Overleay colours 
    label_colors_inst = {
        0: [0, 0, 0],             # Background - black or transparent
        1: [0, 0, 0],             # Nucleus - black or transparent
        2: [255, 105, 180],       # Border - reddish pink
    }

    # Main visualization
    data = next(iter(loader))
    image = data['image'].squeeze(0) # (C, H, W)
    
    mask = data['mask'].squeeze(0).cpu().numpy()  # (H, W) or (H, W, 2) if task == 'panoptic'
    prediction = data['prediction'].squeeze(0).cpu().numpy()  # (H, W)
    # label_colors = label_colors_sem                
    uq_map = data['uq_map'].squeeze(0).cpu().numpy()
    sample_name = data['sample_name'][0]
    
    # Generate colored overlays
    if extra_info['task'].startswith('panoptic'): 
        mask_rgb  = label_to_rgb(inst_to_3c(mask[...,0], False), label_colors_inst, 'instance')
    else:
        mask_rgb = label_to_rgb(mask, sem_maps_colors)
    pred_rgb = label_to_rgb(prediction, sem_maps_colors)
    
    # Create subplots
    fig, axs = plt.subplots(1, 4, figsize=(16, 5))
    titles = ['Input Image', 'Ground Truth', 'Prediction', 'UQ Map']
    overlays = [None, mask_rgb, pred_rgb, uq_map]
    alphas = [1.0, 0.6, 0.6, 0.8]

    for ax, title, overlay, alpha in zip(axs, titles, overlays, alphas):
        if title == 'Input Image':
            ax.imshow(image.permute(1, 2, 0).cpu().numpy())  # RGB input, normalized
        else:
            ax.imshow(image[2], cmap='gray')
            if title in ['Ground Truth', 'Prediction']:
                ax.imshow(overlay, alpha=alpha)
            elif title == 'UQ Map':
                ax.imshow(overlay, cmap='inferno', alpha=alpha)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    fig.suptitle(f"Sample: {sample_name}", fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    output_dir = Path(__file__).parent
    output_file = output_dir / 'sample_batch_overlay_plot.png'
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Overlay plot saved to {output_file}")

if __name__ == "__main__":
    main() 
    