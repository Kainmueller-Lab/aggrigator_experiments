import sys 
import os 
import numpy as np
import json
import torch

from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt

# sys.path.append("C:/Users/cwinklm/Documents/aggrigator_experiments/datasets/")
# sys.path.append('/fast/AG_Kainmueller/vguarin/aggrigator_experiments/')
# print(sys.path)
from datasets.dataset import Dataset_Class

semantic_mapping =  {
    0: 'unlabeled',
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
    38: 'road_2'
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
        self.prediction_path = Path(prediction_path).joinpath('pred_seg')
        self.semantic_mapping_path = semantic_mapping_path
        
        # Extract kwargs with defaults if not provided
        self.task = kwargs.get('task', None)
        self.model_noise = kwargs.get('model_noise', None)
        self.decomp = kwargs.get('decomp', None)
        self.spatial = kwargs.get('spatial', None)
        self.variation = kwargs.get('variation', None)
        self.data_noise = kwargs.get('data_noise', None)
        self.metadata = kwargs.get('metadata', False)
        
        # Uq_map_path and prediction_path parameters must end with the checkpoint folder when passed to the class 
        if not self.uq_map_path.parent.name.startswith('fold0'):
            raise ValueError('Please check the directory for uq_map again.')
        
        if self.data_noise == "1_00":
            self.variation = 'CityScapes'
            if "CityScapes" not in str(self.image_path): 
                raise FileNotFoundError("You are currently using the GTA dataset, the OoD task requires the CityScape dataset. Please adjust paths for images and masks.")

        if self.data_noise == "0_00":
             self.variation = 'GTA'
             if "CityScapes" in str(self.image_path): 
                raise FileNotFoundError("You are currently using the CityScapes dataset, the iD task requires the GTA dataset. Please adjust paths for images and masks.")
        
        self.uq_method = self.__convert_values_uq_name__()[self.uq_map_path.parents[2].name] 
        self.model_ckpt = self.uq_map_path.parent.name
        
        # Complete uq_map directory with either aleaotirc, epistemic or pred_entr folder
        self.uq_map_path = self.uq_map_path.joinpath(self.__convert_values_decomp_name__()[self.decomp])
        
        # Extract the integer indices from filenames
        self.sample_names = [f.split(".")[0] for f in os.listdir(self.uq_map_path) if f.endswith(".tif" )]
    
    def __convert_values_decomp_name__(self):
        return {
            'pu': 'pred_entropy',
            'au': 'aleatoric_uncertainty',
            'eu': 'epistemic_uncertainty',
        }
    
    def __convert_values_uq_name__(self):
        return {
            'Dropout-Final': 'dropout',
            'TTA': 'tta',
            'Ensemble': 'ensemble',
            'Softmax': 'softmax'
        }
        
    def __len__(self):
        """Return the length / number of samples of the dataset."""
        return len(self.sample_names)
    

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
            'decomposition': self.decomp, 
            'metadata': {"model_checkpoint": self.model_ckpt}
        }
        
# ---- Main Function to to test GTA_CityscapesDataset ----   
    
def main():
    extra_info = {
        'task' : 'semantic',
        'variation' : 'GTA',
        'model_noise' : 0,
        'data_noise': '0_00',
        'uq_method' : None,
        'decomp' : 'pu',
        'spatial' : None,
    }

    image_path = "/fast/AG_Kainmueller/data/GTA/OriginalData/preprocessed/images/"
    mask_path = "/fast/AG_Kainmueller/data/GTA/OriginalData/preprocessed/labels/"
    
    uq_map_path = "/fast/AG_Kainmueller/data/GTA_CityScapes_UQ/Dropout-Final/test_results/fold0_seed123/"
    #ood/pred_entropy/"
    
    if extra_info['data_noise'] == "0_00":
        uq_map_path = f"{uq_map_path}/id/"
    else:
        uq_map_path = f"{uq_map_path}/ood/"
        
    prediction_path = uq_map_path
    
    data_loader = GTA_CityscapesDataset(image_path, 
                                  mask_path, 
                                  uq_map_path, 
                                  prediction_path, 
                                  'abc',
                                  **extra_info)
    
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
    
    # # Overleay colours 
    # label_colors_sem = {
    #     0: [0, 0, 0],             # Background - black or transparent
    #     1: [102, 0, 153],         # Epithelial - deep purple
    #     2: [0, 0, 255],           # Plasma Cells - blue
    #     3: [255, 255, 0],         # Lymphocytes - yellow
    #     4: [255, 105, 180],       # Eosinophils - reddish pink
    #     5: [0, 255, 0],           # Fibroblasts - green
    # }
    
    # # Overleay colours 
    # label_colors_inst = {
    #     0: [0, 0, 0],             # Background - black or transparent
    #     1: [255, 105, 180],       # Border - reddish pink
    #     2: [0, 0, 0],             # Nucleus - black or transparent
    # }
    
    # def label_to_rgb(label_map, label_colors):
    #     """Converts a (H, W) label map to an (H, W, 3) RGB overlay."""
    #     h, w = label_map.shape
    #     rgb = np.zeros((h, w, 3), dtype=np.uint8)
    #     for label, color in label_colors.items():
    #         mask = (label_map == label)
    #         rgb[mask] = color
    #     return rgb

    # Main visualization
    data = next(iter(loader))
    image = data['image'].squeeze(0) # (C, H, W)
    
    mask = data['mask'].squeeze(0).cpu().numpy()  # (H, W)
    prediction = data['prediction'].squeeze(0).cpu().numpy()  # (H, W)
    # label_colors = label_colors_sem                
    uq_map = data['uq_map'].squeeze(0).cpu().numpy()
    sample_name = data['sample_name'][0]

    # # Generate colored overlays
    # mask_rgb = label_to_rgb(mask, label_colors)
    # pred_rgb = label_to_rgb(prediction, label_colors)

    print(prediction[0,0,:])
    
    # Create subplots
    fig, axs = plt.subplots(1, 4, figsize=(16, 5))
    titles = ['Input Image', 'Ground Truth', 'Prediction', 'UQ Map']
    overlays = [None, mask, prediction, uq_map]
    alphas = [1.0, 1.0, 1.0, 0.8]

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
    
    
    