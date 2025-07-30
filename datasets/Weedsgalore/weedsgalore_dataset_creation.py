import sys 
import os 
import numpy as np
import mahotas as mh 
import json
import torch

from pathlib import Path
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

import sys
sys.path.append("/fast/AG_Kainmueller/vguarin/aggrigator_experiments/")
from datasets.dataset import Dataset_Class

def inst_to_3c(gt_labels, lizard =  True):
    ''' https://github.com/digitalpathologybern/hover_next_train/blob/main/src/data_utils.py'''
    borders = mh.labeled.borders(gt_labels, Bc=np.ones((3, 3)))
    mask = gt_labels > 0
    if lizard:
        return (((borders & mask) * 1) + (mask * 1))[np.newaxis, :] 
    else:
        return (((borders & mask) * 1) + (mask * 1))

class weedsgalore_dataset(Dataset_Class):
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

        # self.image_path = Path(image_path)
        # self.mask_path = Path(mask_path)
        self.uq_map_path = Path(uq_map_path)
        self.prediction_path = Path(prediction_path)
        self.semantic_mapping_path = Path(semantic_mapping_path)
        
        # Extract kwargs with defaults if not provided
        self.task = kwargs.get('task', None)
        self.model_noise = kwargs.get('model_noise', None)
        self.uq_method = kwargs.get('uq_method', None)
        self.decomp = kwargs.get('decomp', None)
        self.spatial = kwargs.get('spatial', None)
        self.variation = kwargs.get('variation', None)
        self.data_noise = kwargs.get('data_noise', None)
        self.metadata_flag = kwargs.get('metadata', False)
        self.split_path = kwargs.get('split_path', None)
        self.split = kwargs.get('split', ['val_test'])
        if self.split != ['val_test']:
            self.split = ['val_test']
        
        if self.data_noise == '0_00':
            self.uq_map_path = self.uq_map_path.joinpath('rgb_val_test')
            self.prediction_path = self.prediction_path.joinpath('rgb_val_test')
            self.image_path = Path(image_path).joinpath('weedsgalore-dataset')
            self.mask_path = Path(image_path).joinpath('weedsgalore-dataset')
        elif self.data_noise == '1_00':
            self.uq_map_path = self.uq_map_path.joinpath('rgb_ood')
            self.prediction_path = self.prediction_path.joinpath('rgb_ood')
            self.image_path  =  Path(image_path).joinpath('ood_data', 'crops_and_weed_processed')
            self.mask_path  =  Path(image_path).joinpath('ood_data', 'crops_and_weed_processed')
        else:
            raise ValueError('Please, set the noise level to either 0_00 or 1_00 for weedsgalore.')

        if self.metadata_flag is True:
            self.metadata = self.uq_map_path.joinpath(self.task, self.uq_method, "metadata.json")
        self.uq_map_path = self.uq_map_path.joinpath(self.task, self.uq_method, self.decomp)
        self.prediction_path = self.prediction_path.joinpath(self.task, self.uq_method, "pred")
            
        # extract information about the uq maps frorm their location. 
        # uq maps are expected to be saved in the following format: 
        #  "<basefolder>/weedsgalore/<input-type>_<split>/<task>/<uq_methods>/<deomposition>/"
        
        str_idx = str(self.uq_map_path).find("weedsgalore")
        uq_info = str(self.uq_map_path)[str_idx+len("weedsgalore/"):].split("/")
        if len(uq_info[0].split("_")) < 3:
            self.input_type, self.split = uq_info[0].split("_")
        else:
            self.input_type, split_val, split_test = uq_info[0].split("_")
            self.split = f'{split_val}_{split_test}'
        self.in_bands = 3 if self.input_type == "rgb" else 5
        self.num_classes = 6 if self.task == "semantic" else 3

        # Extract the integer indices from filenames
        if self.split_path:
            self.get_sample_names_from_split_file()
        else:
            self.get_sample_names_from_img_directory()
        
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
        if self.data_noise == '0_00':
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
        elif self.data_noise == '1_00':
            img_path = os.path.join(self.image_path, 'images', f'{self.img_list[idx]}.npy')
            img = np.load(img_path)
            img = img.transpose(2,0,1)
        return img

    def get_mask(self, idx):
        """Return the mask at the given index."""
        # load semantic label
        if self.data_noise == '0_00':
            folder = 'semantics' if self.task != 'instance' else 'instances'
            label_path = os.path.join(self.mask_path, self.img_list[idx][:10], folder, self.img_list[idx])
            label = Image.open(label_path + '.png')
            label = np.array(label)

            if self.task == "crops_vs_weed": 
                label[label>1] = 2
        elif self.data_noise == '1_00':
            folder = 'semantics' if self.task != 'crops_vs_weed' else 'crops_vs_weed'
            label_path = os.path.join(self.mask_path, 'masks', folder, f'{self.img_list[idx]}.npy')
            label = np.load(label_path)
        return label 

    def get_uq_map(self, idx):
        """Return the uq_map at the given index."""
        uq_map = np.load(self.uq_map_path.joinpath(f"{self.get_sample_name(idx)}.npy"))
        return uq_map
    
    def get_prediction(self, idx):
        """Return the prediction at the given index."""
        pred = np.load(self.prediction_path.joinpath(f"{self.get_sample_name(idx)}.npy"))      
        return pred.squeeze(0)

    def get_sample_name(self, idx):
        """Return the sample name at the given index."""
        return self.img_list[idx]

    def get_sample_names(self):
        """Return the list of sample names."""
        return self.img_list
    
    def get_sample_names_from_split_file(self):
        """Load sample names from split file."""
        split_path = Path(self.split_path)
        
        with open(split_path, "r") as f:
            self.img_list = json.load(f)  # Assuming the JSON file contains a flat list of filenames
            # self.img_list = np.array(self.img_list) #shall we convert it to numpy array ?
    
    def get_sample_names_from_img_directory(self):
        """Load sample names from directory listing."""
        if self.data_noise == '0_00':
            with open(self.image_path.joinpath('splits', f'{self.split}.txt'), 'r') as file:
                data = [line.rstrip('\n') for line in file]  # Assuming elements are numeric
                self.img_list = np.array(data)
        elif self.data_noise == '1_00':
            self.img_list = sorted(
                [f.stem for f in self.image_path.joinpath('images').iterdir() if f.suffix == '.npy'],
                key=lambda x: (x.split('-')[0], x)
            )

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
            'semantic_mapping': self.get_semantic_mapping() if self.task != "instance" else None,
            'datset_size': len(self),
            'task': self.task,
            'num_classes': self.num_classes,
            'uq_method': self.uq_method,
            'decomposition': self.decomp,
            'metadata': self.metadata,
            'input_typs': self.input_type, 
            'split': self.split 
        }
        return info_dictionary

class OptimizedWeedsGalore(weedsgalore_dataset):
    """Memory-efficient version that can skip loading images"""
    
    def __init__(self, image_path, mask_path, uq_map_path, prediction_path, 
                 semantic_mapping_path, load_images=True, load_preds=True, **kwargs):
        super().__init__(image_path, mask_path, uq_map_path, prediction_path, 
                        semantic_mapping_path, **kwargs)
        self.load_images = load_images
        self.load_preds = load_preds
    
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

class OptimizedWeedsGalore_Properties(weedsgalore_dataset):
    """Memory-efficient version that can skip loading images"""
    
    def __init__(self, image_path, mask_path, uq_map_path, prediction_path, 
                 semantic_mapping_path, load_images=True, load_preds=False, 
                 load_uq_maps=False, **kwargs):
        super().__init__(image_path, mask_path, uq_map_path, prediction_path, 
                        semantic_mapping_path, **kwargs)
        self.load_images = load_images
        self.load_preds = load_preds
        self.load_uq_maps = load_uq_maps
        self.task = 'instance'
    
    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError("Index out of bounds.")
                        
        data = {
            'image': self.get_image(idx),
            'mask': self.get_mask(idx), 
            'sample_name': self.get_sample_name(idx),
        }
        
        # Only load predictions if requested but do not match instance masks
        if self.load_preds:
            data['prediction'] = self.get_prediction(idx)

        # Only load unc. maps if requested but do not match instance masks
        if self.load_uq_maps:
            data['uq_map'] = self.get_uq_map(idx)
        return data
    
# ---- Main Function to to test OptimizedWeedsGalore_Properties ----   
    
def main():
    image_path = "/fast/AG_Kainmueller/data/weedsgalore/"
    mask_path = image_path
    uq_folder =  "/fast/AG_Kainmueller/data/UQ_maps/weedsgalore/"
    pred_folder = uq_folder

    extra_info = {
        'task' : 'crops_vs_weed',
        'variation' : 'maize',
        'model_noise' : 0,
        'data_noise': '1_00',
        'uq_method' : 'dropout',
        'decomp' : 'pu',
        'spatial' : None,
        'metadata' : True,
        'split_path' : None,
        'split' : None
    }
    
    dataset_func = OptimizedWeedsGalore_Properties if extra_info['task'].startswith('istance') else OptimizedWeedsGalore
    
    data_loader = dataset_func(image_path, 
                               mask_path, 
                               uq_folder, 
                               pred_folder, 
                               'abc',
                               load_images=True,
                               **extra_info)
    
    loader = DataLoader(data_loader, 
                        batch_size=1, 
                        shuffle=False,
                        prefetch_factor=2,
                        num_workers=4,
                        pin_memory=True
                        )
    data = next(iter(loader))
    print(data_loader.__len__())
    print(data['image'].shape,
          data['mask'].shape, 
          data['uq_map'].shape, 
          data['prediction'].shape, 
          data['sample_name'])
    
    # Overlay colours 
    label_colors_sem = {
        0: [0, 0, 0],             # Background - black or transparent
        1: [102, 0, 153],         # Crop - deep purple
        2: [0, 255, 0],           # Weed - green
    }
    
    # Overlay colours 
    label_colors_inst = {
        0: [0, 0, 0],             # Background - black or transparent
        1: [0, 0, 0],             # Nucleus - black or transparent
        2: [255, 105, 180],       # Border - reddish pink
    }
    
    def label_to_rgb(label_map, label_colors):
        """Converts a (H, W) label map to an (H, W, 3) RGB overlay."""
        h, w = label_map.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for label, color in label_colors.items():
            mask = (label_map == label)
            rgb[mask] = color
        return rgb

    label_colors = label_colors_sem if data_loader.task != 'instance' else label_colors_inst    
    
    # Main visualization
    loader_iter = iter(loader)      # create an iterator
    index = 50
    for i in range(index):
        next(loader_iter)
        if i == index-1:
            data = next(loader_iter)
    image = data['image'].squeeze(0) # (C, H, W)
    
    mask = data['mask'].squeeze(0).cpu().numpy()  # (H, W)
    prediction = data['prediction'].squeeze(0).cpu().numpy() if data_loader.task != 'instance' else np.zeros_like(mask) # (H, W) 
    uq_map = data['uq_map'].squeeze(0).cpu().numpy() if data_loader.task != 'instance' else np.zeros_like(mask)
    sample_name = data['sample_name'][0]
        
    # Generate colored overlays
    if data_loader.task == 'instance': mask = inst_to_3c(mask, False)
    mask_rgb = label_to_rgb(mask, label_colors)
    pred_rgb = label_to_rgb(prediction, label_colors)
    
    # Create subplots
    fig, axs = plt.subplots(1, 4, figsize=(16, 5))
    
    titles = ['Input Image', 'Ground Truth', 'Prediction', 'UQ Map']
    overlays = [None, mask_rgb, pred_rgb, uq_map]
    alphas = [1.0, 0.8, 0.8, 0.8]

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
    
