import numpy as np
import os
import yaml
import torch
import zarr
import json
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import DataLoader
from datasets.dataset import Dataset_Class

############# Configuration Paths ##############
# ID paths 
# image_path: /fast/AG_Kainmueller/data/data_wormbodies/train
# mask_path: /fast/AG_Kainmueller/data/data_wormbodies/train
# uq_map_path: /fast/AG_Kainmueller/data/UQ_maps/wormbodies/BBBC010_train/fg-bg/dropout/au
# pred_path: /fast/AG_Kainmueller/data/UQ_maps/wormbodies/BBBC010_train/fg-bg/dropout/pred

# OOD paths - v1 
# image_path: /fast/AG_Kainmueller/data/Nematodes/Nematodes/Train_set_processed/resize/images/images_bw_np
# mask_path: /fast/AG_Kainmueller/data/Nematodes/Nematodes/Train_set_processed/resize/masks/binary
# uq_map_path: /fast/AG_Kainmueller/data/UQ_maps/wormbodies/Nematodes_ood/fg-bg/dropout/au
# pred_path: /fast/AG_Kainmueller/data/UQ_maps/wormbodies/Nematodes_ood/fg-bg/dropout/pred

# OOD paths - v2 
# image_path: /fast/AG_Kainmueller/data/Protists/processed/resize/images/images_bw_np
# mask_path: /fast/AG_Kainmueller/data/Protists/processed/resize/masks/binary
# uq_map_path: /fast/AG_Kainmueller/data/UQ_maps/wormbodies/Protists_ood/fg-bg/dropout/au
# pred_path: /fast/AG_Kainmueller/data/UQ_maps/wormbodies/Protists_ood/fg-bg/dropout/pred
##################################################

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
        # self.prediction_path = Path(prediction_path)
        self.semantic_mapping_path = semantic_mapping_path
        self.kwargs = kwargs
        
        # Extract kwargs with defaults if not provided
        self.task = kwargs.get('task', None)
        self.model_noise = kwargs.get('model_noise', None)
        self.uq_method = kwargs.get('uq_method', None)
        self.decomp = kwargs.get('decomp', None)
        self.spatial = kwargs.get('spatial', None)
        self.variation = kwargs.get('variation', None)
        self.data_noise = kwargs.get('data_noise', None)
        self.metadata = kwargs.get('metadata', False)
        self.split_path = kwargs.get('split_path', None)
        self.split = kwargs.get('split', ['test_id'])
        
        if self.data_noise == '0_00':
            self.image_path = self.image_path.joinpath('data_wormbodies', 'train')
            self.mask_path = self.image_path
            self.uq_map_path = self.uq_map_path.joinpath('BBBC010_train')
        elif self.data_noise == '1_00' and self.variation == 'nematodes':
            self.image_path = self.image_path.joinpath('Nematodes', 'Nematodes', 'Train_set_processed', 'resize')
            self.mask_path = self.image_path.joinpath('masks', 'binary')
            self.image_path = self.image_path.joinpath('images', 'images_bw_np')
            self.uq_map_path = self.uq_map_path.joinpath('Nematodes_ood')
        elif self.data_noise == '1_00' and self.variation == 'protists':
            self.image_path = self.image_path.joinpath('Protists', 'processed', 'resize')
            self.mask_path = self.image_path.joinpath('masks', 'binary')
            self.image_path = self.image_path.joinpath('images', 'images_bw_np')
            self.uq_map_path = self.uq_map_path.joinpath('Protists_ood')
        
        self.prediction_path = self.uq_map_path
        self.uq_map_path = self.uq_map_path.joinpath('fg-bg', self.uq_method, self.decomp)
        self.prediction_path = self.prediction_path.joinpath('fg-bg', self.uq_method, 'pred')

        # Extract sample names
        if self.split_path:
            self.get_sample_names_from_split_file()
        else:
            self.get_sample_names_from_img_directory()
        self.file_ending = list(os.listdir(self.image_path))[0].split(".")[1] 

        # f = str(self.uq_map_path)
        # f = f[(f.find("wormbodies")+len("wormbodies/")):].split("/")
        # self.task = f[1]
        # self.uq_method = f[2]
        # self.decomposition = f[3]

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
        img_path = Path(self.image_path).joinpath(f"{self.get_sample_name(idx)}.{self.file_ending}")
        
        if self.file_ending == "zarr":
            zarr_store = zarr.open(img_path, mode='r')
            # crop the image to (512, 512) shape otherwise there are issues with the model output shape 
            image = np.array(zarr_store["volumes"]["raw_bf"][:][0,4:-4, 92:-92].copy())
        elif self.file_ending == "npy":
            image = np.load(img_path)
        return image

    def get_mask(self, idx):
        """Return the mask at the given index."""
        mask_path = Path(self.mask_path).joinpath(f"{self.get_sample_name(idx)}.{self.file_ending}")

        if self.file_ending == "zarr":
            zarr_store = zarr.open(mask_path, mode='r')
            # crop the image to (512, 512) shape otherwise there are issues with the model output shape 
            mask = np.array(zarr_store["volumes"]["gt_fgbg"][:][0,4:-4, 92:-92].copy())
        elif self.file_ending == "npy":
            mask = np.load(mask_path)
        return mask
    
    def get_uq_map(self, idx):
        """Return the uq_map at the given index."""
        uq_map = np.load(self.uq_map_path.joinpath(f"{self.get_sample_name(idx)}.npy"))
        return uq_map
    
    def get_prediction(self, idx):
        """Return the prediction at the given index."""
        pred = np.load(self.prediction_path.joinpath(f"{self.get_sample_name(idx)}.npy"))
        return pred[0, :, :]
    
    def get_sample_names_from_split_file(self):
        """Load sample names from directory listing."""
        split_path = Path(self.split_path)
        
        print(f"Loading sample names from JSON split file: {split_path}")
        with open(split_path, "r") as f:
            names_from_json = json.load(f)  # Assuming the JSON file contains a flat list of filenames
            self.samples = [str(name).split(".")[0] for name in names_from_json] # Ensure we strip any potential file extensions
    
    def get_sample_names_from_img_directory(self):
        """Load sample names from split file."""
        self.samples = [file.split(".")[0] for file in os.listdir(self.image_path)]

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
            'decomposition': self.decomp,
        }
        return info_dictionary
    

def main():
    extra_info = {
        'task' : 'fgbg',
        'variation' : 'nematodes', #'protists', 'nematodes'
        'model_noise' : 0,
        'data_noise': '1_00',
        'uq_method' : 'dropout',
        'decomp' : 'pu',
        'spatial' : None,
        'metadata' : True,
        'split_path' : None,
        'split' : ['test']
    }

    main_folder_name = "UQ_maps" if not extra_info['spatial'] else "UQ_spatial"
    data_path = '/fast/AG_Kainmueller/data/'
    
    # Define the uq_map and prediction paths based on the amsks' noise with which the model was trained        
    uq_map_path = Path(f'/fast/AG_Kainmueller/data/{main_folder_name}/wormbodies/')
    
    data_loader = wormbodies_dataset(data_path, 
                                    data_path, 
                                    uq_map_path, 
                                    uq_map_path, 
                                    'abc',
                                    **extra_info)
    print(data_loader.get_semantic_mapping())
    # print(data_loader.__len__())
    
    loader = DataLoader(data_loader, 
                        batch_size=1, 
                        shuffle=False,
                        prefetch_factor=2,
                        num_workers=4,
                        pin_memory=True
                        )
    iterator = iter(loader)
    batch = next(iterator)
    batch = next(iterator)
    data = next(iterator)
    print(data['image'].shape,
          data['mask'].shape, #if task == 'instance', then mask[...,0] is instances and mask[...,1] is 3-class instance
          data['uq_map'].shape, #if task == 'instance', then uq_map[...,0] is 3-class instance
          data['prediction'].shape, #if task == 'instance', then uq_map[...,0] is instances and uq_map[...,1] is 3-class instance
          data['sample_name'])
    
    # Assuming batch size B=1, squeeze to get rid of batch dimension
    image = data['image'].squeeze(0)  # Shape: C x H x W
    mask = data['mask'].squeeze(0)    # Shape: H x W
    uq_map = data['uq_map'].squeeze(0)  # Shape: H x W
    prediction = data['prediction'].squeeze(0)  # Shape: H x W
    sample_name = data['sample_name'][0]  # Assuming it's a list of strings
    
    # Create subplots
    fig, axs = plt.subplots(1, 4, figsize=(16, 5))
    titles = ['Input Image', 'Ground Truth', 'Prediction', 'UQ Map']
    overlays = [None, mask, prediction, uq_map]
    cmaps = [None, 'Purples', 'Purples', 'inferno']
    alphas = [1.0, 0.6, 0.6, 0.8]  # transparency for overlays

    for ax, title, overlay, cmap, alpha in zip(axs, titles, overlays, cmaps, alphas):
        ax.imshow(image, cmap='gray')
        if overlay is not None:
            ax.imshow(overlay, cmap=cmap, alpha=alpha)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    # Add sample name as the overall title
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


