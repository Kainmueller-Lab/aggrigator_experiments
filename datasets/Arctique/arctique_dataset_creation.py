import torch
import os 
import numpy as np
import mahotas as mh
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import rgb_to_grayscale
from pathlib import Path
from PIL import Image
from functools import lru_cache

from datasets.dataset import Dataset_Class

# ---- Arctique Config. Functions ----

# cell IDS from https://zenodo.org/records/14016860
mapping_dict = {
            0: "Background",
            1: "Epithelial",
            2: "Plasma Cells",
            3: "Lymphocytes",
            4: "Eosinophils",
            5: "Fibroblasts",
        }

# ---- Arctique Dataset Creation Functions ----

def inst_to_3c(gt_labels, lizard =  True):
    ''' https://github.com/digitalpathologybern/hover_next_train/blob/main/src/data_utils.py'''
    borders = mh.labeled.borders(gt_labels, Bc=np.ones((3, 3)))
    mask = gt_labels > 0
    if lizard:
        return (((borders & mask) * 1) + (mask * 1))[np.newaxis, :] 
    else:
        return (((borders & mask) * 1) + (mask * 1))
    
@staticmethod
def normalize_min_max(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if mi is None:
        mi = np.min(x)
    if ma is None:
        ma = np.max(x)
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)
    return x

class renderHE_UQ_HVNext(Dataset): 
    def __init__(self, root_dir, mode, image_noise = "0_00"): #mode is val, test
        self.root_dir = root_dir
        self.mode = mode
        self.image_noise = image_noise 

        hovernext_masks_path = root_dir 
        
        self.image_dir = Path(root_dir).joinpath(f"{image_noise}/images")
        self.inst_mask_dir = Path(hovernext_masks_path).joinpath(f"{image_noise}/masks/instance_indexing") #.joinpath(self.mode).joinpath(f"masks/instance_indexing") prior to v1-0-corrected
        self.sem_mask_dir = Path(hovernext_masks_path).joinpath(f"{image_noise}/masks/semantic_indexing") #.joinpath(self.mode).joinpath(f"masks/semantic_indexing") prior to v1-0-corrected
    
        # extract the integer indices from filenames    
        self.sample_names = [int(digits) for filename in os.listdir(self.image_dir) 
                             if (digits := ''.join(filter(str.isdigit, filename)))
                             ]

    def __len__(self):
        return len(self.sample_names)

    def get_samplename(self, idx):
        return self.sample_names[idx]

    def __getitem__(self, idx):
        fn = self.get_samplename(idx)

        image_file = self.image_dir.joinpath(f"img_{fn}.png")
        image = np.array(Image.open(image_file)).astype(np.float32)
        image = image[:, :, :3] # remove alpha channel 

        if self.mode == 'test': 
            inst_file = self.inst_mask_dir.joinpath(f"{fn}.tif")
            sem_file = self.sem_mask_dir.joinpath(f"{fn}.tif") 
        
        inst_label = np.array(Image.open(inst_file), dtype = int)
        sem_label = np.array(Image.open(sem_file), dtype = int)

        label_3c = inst_to_3c(inst_label, False)
        label = np.stack((inst_label, sem_label, label_3c), axis=-1)
        label = torch.tensor(label, dtype=torch.long)

        image = normalize_min_max(image, 0, 255)
        image = torch.tensor(image, dtype=torch.float32) 
        image  = image.permute(2, 0, 1) # CxHxW  
        return image, label 
    
# ---- Arctique Dataset Creation Functions Based on Abstract Class ----

class ArctiqueDataset(Dataset_Class):
    def __init__(self, image_path, mask_path, uq_map_path, prediction_path, semantic_mapping_path, **kwargs: dict):
        super().__init__(image_path, mask_path, uq_map_path, prediction_path, semantic_mapping_path, **kwargs)
  
        self.image_path = Path(image_path)
        self.uq_map_path = Path(uq_map_path)
        self.prediction_path = Path(prediction_path)
        self.semantic_mapping_path = semantic_mapping_path
                
        self.inst_mask_dir = Path(mask_path).joinpath("instance_indexing")
        self.sem_mask_dir = Path(mask_path).joinpath("semantic_indexing") 
        
        # extract the integer indices from filenames    
        self.sample_names = [int(digits) for filename in os.listdir(self.image_path) 
                             if (digits := ''.join(filter(str.isdigit, filename)))
                             ]
        
        # Extract kwargs with defaults if not provided
        self.task = kwargs.get('task', None)
        self.model_noise = kwargs.get('model_noise', None)
        self.uq_method = kwargs.get('uq_method', None)
        self.decomp = kwargs.get('decomp', None)
        self.spatial = kwargs.get('spatial', None)
        self.variation = kwargs.get('variation', None)
        self.data_noise = kwargs.get('data_noise', None)
        self.metadata = kwargs.get('metadata', False)
        
        # Load and cache metadata indices if metadata validation is enabled
        self._metadata_indices = None
        if self.metadata:
            self._load_metadata_indices()
        
    def _load_metadata_indices(self):
        """Load metadata indices once and cache them"""
        parent_path = self.uq_map_path.parent
        self.metadata_path = parent_path.joinpath('UQ_metadata')
        metadata_file = f'{self.task}_noise_{self.model_noise}_{self.variation}_{self.data_noise}_{self.uq_method}_pu_sample_idx.npy'
        self._metadata_indices = np.load(self.metadata_path.joinpath(metadata_file))
        
    def _get_metadata_index(self, sample_name):
        """Get the index position of sample_name in metadata array"""
        if self._metadata_indices is None:
            raise ValueError("Metadata not loaded. Set metadata=True in constructor.")
        # Find where sample_name appears in metadata_indices
        try:
            metadata_idx = np.where(self._metadata_indices == sample_name)[0][0]
            return metadata_idx
        except IndexError:
            raise ValueError(f"Sample name {sample_name} not found in metadata indices")
        
    def __len__(self):
        return len(self.sample_names)
    
    def __getitem__(self, idx):
        
        if idx >= self.__len__():
            raise IndexError("Index out of bounds.")
                
        data = {
            'image': self.get_image(idx),
            'mask': self.get_mask(idx),
            'uq_map': self.get_uq_map(idx),
            'prediction': self.get_prediction(idx),
            'sample_name': self.get_sample_name(idx),
        }
        
        return data
    
    def get_image(self, idx):
        fn = self.get_sample_name(idx)
        
        image_file = self.image_path.joinpath(f"img_{fn}.png")
        image = np.array(Image.open(image_file)).astype(np.float32)
        image = image[:, :, :3] # remove alpha channel 
        image = normalize_min_max(image, 0, 255)
        image = torch.tensor(image, dtype=torch.float32) 
        image  = image.permute(2, 0, 1) # CxHxW  
        return image
    
    def get_mask(self, idx):
        fn = self.get_sample_name(idx)
        
        inst_file = self.inst_mask_dir.joinpath(f"{fn}.tif")
        sem_file = self.sem_mask_dir.joinpath(f"{fn}.tif") 
        
        inst_label = np.array(Image.open(inst_file), dtype = int)
        sem_label = np.array(Image.open(sem_file), dtype = int)
        three_label = inst_to_3c(inst_label, False)
        
        if self.task.startswith('semantic'):
            return torch.tensor(sem_label, dtype=torch.long)
        elif self.task.startswith('instance'):
            label = np.stack((inst_label, three_label), axis=-1)
            return torch.tensor(label, dtype=torch.long)
        
        label = np.stack((inst_label, sem_label, three_label), axis=-1)
        return torch.tensor(label, dtype=torch.long)
    
    @lru_cache(maxsize=32)
    def get_uq_map(self, idx):
        """Load uncertainty maps"""
        sample_name = self.get_sample_name(idx)
        if self.metadata:
            fn = self._get_metadata_index(sample_name)  # Use metadata index to get correct position in uq_map array
        else:
            fn = idx  # Fallback: assume sample_names order matches array order
        
        map_type_sem = f"semantic_noise_{self.model_noise}_{self.variation}_{self.data_noise}_{self.uq_method}_{self.decomp}"
        map_type_threeinst = f"instance_noise_{self.model_noise}_{self.variation}_{self.data_noise}_{self.uq_method}_{self.decomp}"

        if self.spatial:
            map_type_sem += f'_{self.spatial}'
            map_type_threeinst += f'_{self.spatial}'
        map_type_sem += ".npy"
        map_type_threeinst += ".npy"
        
        map_file_sem = self.uq_map_path .joinpath(map_type_sem)
        map_file_inst = self.uq_map_path .joinpath(map_type_threeinst)
        
        if self.task.startswith('semantic'):
            return np.load(map_file_sem)[fn]
        elif self.task.startswith('instance'):
            return np.load(map_file_inst)[fn]
        
        return np.stack((np.load(map_file_inst)[fn],np.load(map_file_inst)[fn]), axis=-1)
    
    def get_prediction(self, idx, **kwargs):
        """Load panoptic model predictions"""
        sample_name = self.get_sample_name(idx)
        if self.metadata:
            fn = self._get_metadata_index(sample_name)  # Use metadata index to get correct position in uq_map array
        else:
            fn = idx  # Fallback: assume sample_names order matches array order
                
        preds_inst_type = f"instance_noise_{self.model_noise}_{self.variation}_{self.data_noise}_{self.uq_method}.npy"
        preds_sem_type = f"semantic_noise_{self.model_noise}_{self.variation}_{self.data_noise}_{self.uq_method}.npy"
        
        preds_file_path_inst = self.prediction_path.joinpath(preds_inst_type)
        preds_file_path_sem = self.prediction_path.joinpath(preds_sem_type)
        
        preds_inst, preds_sem = np.load(preds_file_path_inst)[fn], np.load(preds_file_path_sem)[fn]
        three_preds = inst_to_3c(preds_inst, False)
        
        if self.task.startswith('semantic'):
            return preds_sem
        elif self.task.startswith('instance'):
            preds_inst = np.stack((preds_inst, three_preds), axis=-1)
            return preds_inst
        return np.stack((preds_inst, preds_sem, three_preds), axis=-1) 
    
    def get_sample_name(self, idx):
        """Return the sample name at the given index."""
        return self.sample_names[idx]
             
    def get_sample_names(self):
        return self.sample_names
    
    def get_semantic_mapping(self):
        return mapping_dict
    
    def get_info(self):
        """Return a dictionary with information about the dataset."""
        info_dictionary = {
            'image_path': str(self.image_path),
            'mask_path': str(self.mask_path),
            'uq_map_path': str(self.uq_map_path),
            'prediction_path': str(self.prediction_path),
            'semantic_mapping': self.get_semantic_mapping(), 
            'dataset_size': len(self),  # Fixed typo: datset_size -> dataset_size
            'task': self.task,
            'num_classes': None,  # TODO: Implement based on your data
            'uq_method': self.uq_method,
            'decomposition': self.decomp,
            # 'metadata': None  # TODO: Add metadata if available
        }
        return info_dictionary

def main():
    extra_info = {
        'task' : 'semantic',
        'variation' : 'blood_cells',
        'model_noise' : 0,
        'data_noise': '0_25',
        'uq_method' : 'dropout',
        'decomp' : 'pu',
        'spatial' : 'high_moran',
        'metadata' : True,
    }
    
    main_folder_name = "UQ_maps" if not extra_info['spatial'] else "UQ_spatial"
    map_path = Path('/fast/AG_Kainmueller/vguarin/hovernext_trained_models/trained_on_cluster/uncertainty_arctique_v1-0-corrected_14')
    base_path = Path('/fast/AG_Kainmueller/synth_unc_models/data/v1-0-variations/variations/')
    
    image_path = base_path.joinpath(extra_info['variation'], extra_info['data_noise'], 'images')
    mask_path = base_path.joinpath(extra_info['variation'], extra_info['data_noise'], 'masks')
    prediction_path = map_path.joinpath('UQ_predictions')
    uq_map_path = map_path.joinpath(main_folder_name)
    
    data_loader = ArctiqueDataset(image_path, 
                                  mask_path, 
                                  uq_map_path, 
                                  prediction_path, 
                                  'abc',
                                  **extra_info)
    print(data_loader.get_semantic_mapping())
    
    loader = DataLoader(data_loader, 
                        batch_size=1, 
                        shuffle=False,
                        prefetch_factor=2,
                        num_workers=4,
                        pin_memory=True
                        )
    data = next(iter(loader))
    print(data['image'].shape,
          data['mask'].shape, #if task == 'instance', then mask[...,0] is instances and mask[...,1] is 3-class instance
          data['uq_map'].shape, #if task == 'instance', then uq_map[...,0] is instances and uq_map[...,1] is 3-class instance
          data['prediction'].shape, #if task == 'instance', then uq_map[...,0] is instances and uq_map[...,1] is 3-class instance
          data['sample_name'])
    
    # Overleay colours 
    label_colors_sem = {
        0: [0, 0, 0],             # Background - black or transparent
        1: [102, 0, 153],         # Epithelial - deep purple
        2: [0, 0, 255],           # Plasma Cells - blue
        3: [255, 255, 0],         # Lymphocytes - yellow
        4: [255, 105, 180],       # Eosinophils - reddish pink
        5: [0, 255, 0],           # Fibroblasts - green
    }
    
    # Overleay colours 
    label_colors_inst = {
        0: [0, 0, 0],             # Background - black or transparent
        1: [255, 105, 180],       # Border - reddish pink
        2: [0, 0, 0],             # Nucleus - black or transparent
    }
    
    def label_to_rgb(label_map, label_colors):
        """Converts a (H, W) label map to an (H, W, 3) RGB overlay."""
        h, w = label_map.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for label, color in label_colors.items():
            mask = (label_map == label)
            rgb[mask] = color
        return rgb

    # Main visualization
    data = next(iter(loader))
    image = data['image'].squeeze(0) # (C, H, W)
    
    if extra_info['task'].startswith('semantic'):
        mask = data['mask'].squeeze(0).cpu().numpy()  # (H, W)
        prediction = data['prediction'].squeeze(0).cpu().numpy()  # (H, W)
        label_colors = label_colors_sem        
    else:
        mask = data['mask'][...,1].squeeze(0).cpu().numpy() # (H, W) 
        prediction = data['prediction'][...,1].squeeze(0).cpu().numpy()  # (H, W)
        label_colors = label_colors_inst
        
    uq_map = data['uq_map'].squeeze(0)
    sample_name = data['sample_name'][0]

    # Generate colored overlays
    mask_rgb = label_to_rgb(mask, label_colors)
    pred_rgb = label_to_rgb(prediction, label_colors)

    # Create subplots
    fig, axs = plt.subplots(1, 4, figsize=(16, 5))
    titles = ['Input Image', 'Ground Truth', 'Prediction', 'UQ Map']
    overlays = [None, mask_rgb, pred_rgb, uq_map.cpu().numpy()]
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


        
    
    
        
        
    
    
         
