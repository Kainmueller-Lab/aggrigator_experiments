import sys 
import os 
import numpy as np

from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt

# sys.path.append("C:/Users/cwinklm/Documents/aggrigator_experiments/datasets/")
# sys.path.append('/fast/AG_Kainmueller/vguarin/aggrigator_experiments/')
# print(sys.path)
from datasets.dataset import Dataset_Class
from .cityscapes_labels import labels, trainId2label #from cityscapes_labels if you directly run this script !

# Extract mappings from the labels
color2trainId = {}
trainId2color = {}
semantic_mapping = {}

for label in labels:
    # Build color to trainId mapping
    color2trainId[label.color] = label.trainId
    
    # Build trainId to color mapping (skip ignore class 255 duplicates)
    if label.trainId not in trainId2color or label.trainId != 255:
        trainId2color[label.trainId] = label.color
    
    # Build semantic mapping (trainId -> name)
    # Only include valid trainIds (not 255 which is ignore)
    if label.trainId != 255:
        semantic_mapping[label.trainId] = label.name

# Add ignore class to semantic mapping
semantic_mapping[255] = 'ignore'

print(f"Loaded {len([l for l in labels if l.trainId != 255])} semantic classes")
print(f"Color mapping contains {len(color2trainId)} colors")
print(f"Valid trainIDs: {sorted([tid for tid in semantic_mapping.keys() if tid != 255])}")

new_semantic_mapping =  {
    0: ['road', (128, 64, 128)],
    1: ['sidewalk', (244, 35, 232)],
    2: ['building', (70, 70, 70)],
    3: ['wall', (102, 102, 156)],
    4: ['fence', (190, 153, 153)],
    5: ['pole', (153, 153, 153)],
    6: ['traffic light', (250, 170, 30)],
    7: ['traffic sign', (220, 220, 0)],
    8: ['vegetation', (107, 142, 35)],
    9: ['terrain', (152, 251, 152)],
    10: ['sky', (70, 130, 180)],
    11: ['person', (220, 20, 60)],
    12: ['rider', (255, 0, 0)],
    13: ['car', (0, 0, 142)],
    14: ['truck', (0, 0, 70)],
    15: ['bus',  (0, 60, 100)],
    16: ['train', (0, 80, 100)],
    17: ['motorcycle', (0, 0, 230)],
    18: ['bicycle', (119, 11, 32)],
    19: ['sidewalk_2', (46, 247, 180)],
    20: ['person_2', (167, 242, 242)],
    21: ['car_2', (30, 193, 252)],
    22: ['vegetation_2', (242, 160, 19)],
    23: ['road_2', (84, 86, 22)]
}

class GTA_CityscapesDataset(Dataset_Class):
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
        
        self.image_path = Path(image_path)
        self.mask_path = Path(mask_path)
        self.uq_map_path = Path(uq_map_path)
        self.semantic_mapping_path = semantic_mapping_path
        self.prediction_path = Path(prediction_path)
        
        # Extract kwargs with defaults if not provided
        self.task = kwargs.get('task', None)
        self.uq_method = kwargs.get('uq_method', None)
        self.model_noise = kwargs.get('model_noise', None)
        self.decomp = kwargs.get('decomp', None)
        self.spatial = kwargs.get('spatial', None)
        self.variation = kwargs.get('variation', None)
        self.data_noise = kwargs.get('data_noise', None)
        self.metadata = kwargs.get('metadata', False)
        self.split_path = kwargs.get('split_path', None)
        self.split = kwargs.get('split', ['test'])
        
        # Validate required parameters
        self.__validate_required_params__()
              
        # Set up dataset-specific paths and configurations
        self.__setup_dataset_paths__()
        
        # Extract the integer indices from filenames
        if self.split_path:
            self.get_sample_names_from_split_file()
        else:
            self.get_sample_names_from_uq_directory()
    
    def __convert_to_values_decomp_name__(self):
        return {
            'pu': 'pred_entropy',
            'au': 'aleatoric_uncertainty',
            'eu': 'epistemic_uncertainty',
        }
    
    def __convert_to_values_uq_name__(self):
        return {
            'dropout': 'Dropout-Final',
            'tta': 'TTA',
            'ensemble': 'Ensemble',
            'softmax': 'Softmax'
        }
        
    def __setup_dataset_paths__(self):
        """Set up dataset-specific paths and validate dataset consistency."""
        # Set variation based on data_noise
        if self.data_noise in ["0_00", "1_00"] and not self.variation:
            self.variation = 'cityscapes'
        
        # Validate dataset consistency
        self. __validate_dataset_consistency__()
        
        # Build folder following base path for uq_maps and uq_predictions
        uq_map_base = self.uq_map_path.joinpath(
            self.__convert_to_values_uq_name__()[self.uq_method], 
            'test_results', 
            'fold0_seed123'
        )
        
        # Uq_map_path and prediction_path parameters must end with the checkpoint folder when passed to the class 
        if not uq_map_base.name.startswith('fold0'):
            raise ValueError(f"Invalid directory structure. Expected folder starting with 'fold0', got: {uq_map_base.name}")
        
        # Set task-specific paths
        if self.data_noise == "0_00":
            self.uq_map_path = uq_map_base / 'id'
        else:
            self.uq_map_path = uq_map_base / 'ood'
        
        # Set prediction path
        self.prediction_path = self.uq_map_path.joinpath('pred_seg')
        
        # Extract model checkpoint name; previously: self.uq_method = self.__convert_to_values_uq_name__()[self.uq_map_path.parents[2].name] 
        self.model_ckpt = self.uq_map_path.parent.name
        
        # Complete uq_map directory with either decomposition type: aleatoric, epistemic or pred_entr
        self.uq_map_path = self.uq_map_path.joinpath(self.__convert_to_values_decomp_name__()[self.decomp])
        
        # Final validation that paths exist
        if not self.uq_map_path.exists():
            raise FileNotFoundError(f"Uncertainty map path does not exist: {self.uq_map_path}")
    
    def __validate_required_params__(self):
        """Validate that required parameters are provided."""
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
        is_cityscapes = "CityScapes" in str(self.image_path)
        
        if self.data_noise == "1_00":  # OoD task
            if not is_cityscapes:
                raise FileNotFoundError(
                    "OoD task (data_noise='1_00') requires CityScapes dataset. "
                    f"Current image path: {self.image_path}"
                )
        
        elif self.data_noise == "0_00":  # iD task
            if is_cityscapes:
                raise FileNotFoundError(
                    "iD task (data_noise='0_00') requires GTA dataset. "
                    f"Current image path: {self.image_path}"
                )
                
    def __len__(self):
        """Return the length / number of samples of the dataset."""
        return len(self.sample_names) 
    

    def __getitem__(self, idx):
        """Return a dictionary with sample at given index. """

        # Get prediction as RGB colors first
        pred_colors = self.get_prediction_colors(idx)
                
        sample = {
            'image': self.get_image(idx),
            'mask': self.get_mask(idx),
            'uq_map': self.get_uq_map(idx),
            'prediction': self.rgb_to_trainid(pred_colors),  # Convert RGB prediction to trainID
            'pred_colors': self.get_prediction_colors(idx),  # RGB colors for visualization [H, W, 3]
            'sample_name': self.get_sample_name(idx)
        }

        return sample
    
    def rgb_to_trainid(self, rgb_image):
        """Convert RGB prediction image to trainID semantic segmentation mask."""
        h, w = rgb_image.shape[:2]
        trainid_mask = np.full((h, w), 255, dtype=np.uint8)  # Default to ignore class
        
        # Convert RGB to trainID using color mapping from cityscapes labels
        for color, trainid in color2trainId.items():
            # Find pixels matching this color (exact match)
            mask = np.all(rgb_image == color, axis=-1)
            if np.any(mask):  # Only update if pixels found
                trainid_mask[mask] = trainid
        
        # Report unknown colors for debugging
        unique_colors = np.unique(rgb_image.reshape(-1, 3), axis=0)
        unknown_colors = []
        for color in unique_colors:
            color_tuple = tuple(color)
            if color_tuple not in color2trainId:
                unknown_colors.append(color_tuple)
        
        if unknown_colors:
            print(f"Warning: Found {len(unknown_colors)} unknown colors in prediction: {unknown_colors[:5]}...")
            
        return trainid_mask
    
    def get_image(self, idx):
        """Return the image at the given index."""
        filename = self.sample_names[idx] + ".npy"
        img = np.load(self.image_path.joinpath(filename))
        img = img.transpose(2,0,1)
        return img

    def get_mask(self, idx):
        """Return the mask at the given index."""
        filename = self.sample_names[idx] + ".npy"
        mask = np.load(self.mask_path.joinpath(filename))
        return mask
    
    def get_uq_map(self, idx):
        """Return the uq_map at the given index."""
        filename = self.sample_names[idx] + ".tif"
        uq_map = np.array(Image.open(self.uq_map_path.joinpath(filename)))
        return uq_map
    
    def get_prediction_colors(self, idx):
        """Return the prediction colors (RGB) at the given index."""
        filename = self.sample_names[idx] + "_mean" + ".png"
        pred_colors = np.array(Image.open(self.prediction_path.joinpath(filename)))
        return pred_colors
    
    def get_prediction(self, idx):
        """Return the prediction at the given index."""
        filename = self.sample_names[idx] + "_mean" + ".png"
        pred = np.array(Image.open(self.prediction_path.joinpath(filename)))
        return pred

    def get_sample_name(self, idx):
        """Return the sample name at the given index."""
        return self.sample_names[idx]

    def get_sample_names(self):
        """Return the list of sample names."""
        return self.sample_names
    
    def get_sample_names_from_split_file(self):
        """Load sample names from split file."""
        split_path = Path(self.split_path)
        
        with open(split_path, "r") as f:
            self.sample_names = [
                line.strip().split(".")[0] 
                for line in f 
                if line.strip().endswith(".tif")
            ]
    
    def get_sample_names_from_uq_directory(self):
        """Load sample names from directory listing."""
        self.sample_names = [
            f.split(".")[0] 
            for f in os.listdir(self.uq_map_path) 
            if f.endswith(".tif")
        ]
    
    def get_semantic_mapping(self):
        """Return the semantic mapping dictionary (trainID -> [class name, color])."""
        return new_semantic_mapping
    
    def get_trainid_to_color_mapping(self):
        """Return the trainID to color mapping for visualization."""
        return trainId2color
    
    def get_color_to_trainid_mapping(self):
        """Return the color to trainID mapping for conversion."""
        return color2trainId
    
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
            'num_classes': len([tid for tid in trainId2color.keys() if tid != 255]),
            'semantic_mapping': self.get_semantic_mapping(),
            'uq_method': self.uq_method, 
            'decomposition': self.decomp, 
            'metadata': {"model_checkpoint": self.model_ckpt}
        }
        return info_dictionary

class OptimizedGTA_CityscapesDataset(GTA_CityscapesDataset):
    """Memory-efficient version that can skip loading images"""
    
    def __init__(self, image_path, mask_path, uq_map_path, prediction_path, 
                 semantic_mapping_path, load_images=False, load_preds=False, 
                 max_samples=500, **kwargs):
        super().__init__(image_path, mask_path, uq_map_path, prediction_path, 
                        semantic_mapping_path, **kwargs)
        self.load_images = load_images
        self.load_preds = load_preds
        # Limit the number of samples if specified
        if max_samples is not None and max_samples < len(self.sample_names):
            self.sample_names = self.sample_names[:max_samples]
        
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
    extra_info = {
        'task' : 'semantic',
        'variation' : 'cityscapes',
        'model_noise' : 0,
        'data_noise': '1_00',
        'uq_method': 'dropout',
        'decomp' : 'pu',
        'spatial' : None,
        'split_path' : None,
        'split' : None
    }

    base_path = "/fast/AG_Kainmueller/data"
    data_folder_name = "/GTA/CityScapesOriginalData" # /GTA/CityScapesOriginalData
    
    if data_folder_name.startswith('/GTA/City'):
        splits_folder = 'Cityscapes_ood'
        
    else:
        splits_folder = 'GTA_id_test'
    
    image_path = f"{base_path}/{data_folder_name}/preprocessed/images/"
    mask_path = f"{base_path}/{data_folder_name}/preprocessed/labels/"
    uq_map_path = f"{base_path}/GTA_CityScapes_UQ/"
    prediction_path = uq_map_path
    
    text_path = f"{base_path}/GTA_ValUES_splits/{splits_folder}"
    extra_info['split_path'] = text_path
    
    data_loader = GTA_CityscapesDataset(image_path, 
                                  mask_path, 
                                  uq_map_path, 
                                  prediction_path, 
                                  'abc',
                                  **extra_info)
    sem_maps_colors = data_loader.get_semantic_mapping()
    
    loader = DataLoader(data_loader, 
                        batch_size=1, 
                        shuffle=False,
                        prefetch_factor=2,
                        num_workers=4,
                        pin_memory=True
                        )
    data = next(iter(loader))
    print(data['image'].shape,
          data['mask'].shape, 
          data['uq_map'].shape, 
          data['prediction'].shape, 
          data['sample_name'])
    
    def label_to_rgb(label_map, label_colors):
        """Converts a (H, W) label map to an (H, W, 3) RGB overlay."""
        h, w = label_map.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for label, color in label_colors.items():
            mask = (label_map == label)
            rgb[mask] = color[1]
        return rgb

    # Main visualization
    data = next(iter(loader))
    image = data['image'].squeeze(0) # (C, H, W)
    
    mask = data['mask'].squeeze(0).cpu().numpy()  # (H, W)
    prediction = data['prediction'].squeeze(0).cpu().numpy()  # (H, W)
    # label_colors = label_colors_sem                
    uq_map = data['uq_map'].squeeze(0).cpu().numpy()
    sample_name = data['sample_name'][0]
    
    # Generate colored overlays
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
    
    
    