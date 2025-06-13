import os
import numpy as np
import torch
import nibabel as nib
import glob
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from functools import lru_cache
from typing import Optional, Tuple

from datasets.dataset import Dataset_Class

# ---- Arctique Config. Functions ----

# cell IDS from https://www.cancerimagingarchive.net/collection/lidc-idri/ 
mapping_dict = {
            0: "Background",
            1: "Lung Nodule",
        }

# ---- LIDC_IDRI Dataset Creation Functions ----

# @staticmethod
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

class LIDC_UQ_Dataset(Dataset):
    def __init__(self, root_dir, mode='test', OOD = False, normalize_input=True, 
                 consensus_threshold=2, return_individual_masks=False, return_2d_slices = True):
        """
        LIDC-IDRI Dataset loader with multi-rater consensus
        
        Args:
            root_dir: Path to the dataset folder (e.g., .../malignancy_fold0_seed123/)
            mode: Dataset mode ('test', 'val', etc.)
            image_noise: Noise level folder name (e.g., "0_00")  
            normalize_input: Whether to normalize input images to [0,1]
            consensus_threshold: Minimum number of raters that must agree (1-4)
            return_individual_masks: Whether to return individual rater masks
        """
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.ood = OOD
        self.image_noise = "0_00" if self.ood == False else "1_00"
        self.normalize_input = normalize_input
        self.consensus_threshold = consensus_threshold
        self.return_individual_masks = return_individual_masks
        self.return_2d_slices = return_2d_slices
        
        # Set up paths based on folder structure
        if self.image_noise == "0_00":
            self.data_dir = self.root_dir / "id"
        else:
            self.data_dir = self.root_dir / "ood"
            
        self.input_dir = self.data_dir / "input"
        self.gt_seg_dir = self.data_dir / "gt_seg"
        
        # Get sample names by looking for input files
        self.sample_names = self._get_sample_names()
        
    def _get_sample_names(self):
        """Extract unique sample names from input directory"""
        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
            
        input_files = list(self.input_dir.glob("*.nii.gz"))
        if not input_files:
            raise ValueError(f"No .nii.gz files found in {self.input_dir}")
            
        # Extract base sample names (e.g., "0103_00" from "0103_00.nii.gz")
        self.sample_names = []
        for file in input_files:
            sample_name = file.stem.replace('.nii', '')  # Remove .nii from .nii.gz
            self.sample_names.append(sample_name)
            
        return sorted(list(set(self.sample_names)))
        # print(f"Found {len(self.sample_names)} samples: {self.sample_names[:5]}{'...' if len(self.sample_names) > 5 else ''}")
        
    def _load_nifti(self, file_path):
        """Load NIfTI file and return data"""
        try:
            img = nib.load(str(file_path))
            data = img.get_fdata()
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
            
    def _create_consensus_mask(self, sample_name):
        """Create consensus mask from multiple rater annotations"""
        gt_masks = []
        # Look for rater masks (0103_00_00, 0103_00_01, etc.)
        for rater_id in range(4):  # 4 raters (00, 01, 02, 03)
            mask_file = self.gt_seg_dir / f"{sample_name}_{rater_id:02d}.nii.gz"
            
            if mask_file.exists():
                mask_data = self._load_nifti(mask_file)
                if mask_data is not None:
                    gt_masks.append(mask_data)
                else:
                    print(f"Warning: Could not load {mask_file}")
            else:
                print(f"Warning: Rater mask not found: {mask_file}")
                
        if not gt_masks:
            raise ValueError(f"No valid rater masks found for sample {sample_name}")
            
        # Stack all rater masks and create consensus
        gt_stack = np.stack(gt_masks, axis=-1)  # Shape: (H, W, D, num_raters)
        # Create consensus based on threshold
        if self.consensus_threshold == 1: #ValUES consensus
            # Any rater agrees (union)
            consensus_mask = np.any(gt_stack, axis=-1).astype(np.uint8)
        else:
            # At least N raters agree
            consensus_mask = (np.sum(gt_stack, axis=-1) >= self.consensus_threshold).astype(np.uint8)
            
        if self.return_individual_masks:
            return consensus_mask, gt_stack
        else:
            return consensus_mask
            
    def __len__(self):
        return len(self.sample_names)
        
    def get_samplename(self, idx):
        return self.sample_names[idx]
        
    def __getitem__(self, idx):
        sample_name = self.get_samplename(idx)
        
        # Load input image
        input_file = self.input_dir / f"{sample_name}.nii.gz"
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        image = self._load_nifti(input_file)
        if image is None:
            raise ValueError(f"Could not load input image: {input_file}")
            
        # Load and create consensus mask
        if self.return_individual_masks:
            consensus_mask, individual_masks = self._create_consensus_mask(sample_name)
        else:
            consensus_mask = self._create_consensus_mask(sample_name)
            
        # Normalize input image if requested
        if self.normalize_input:
            # For medical images, typically normalize to [0,1] based on data range
            image = normalize_min_max(image, None, None, clip=True)
        
        # Convert to tensors
        image = torch.tensor(image, dtype=torch.float32)
        consensus_mask = torch.tensor(consensus_mask, dtype=torch.long)
        
        # Add channel dimension if needed (for 3D: C x H x W x D, for 2D: C x H x W)
        # if len(image.shape) == 3 and len(consensus_mask.shape) == 3:
        #     image = image.unsqueeze(0)  # Add channel dimension
        #     consensus_mask = consensus_mask.unsqueeze(0)
        # elif len(image.shape) == 2 and len(consensus_mask.shape) == 2:
        #     image = image.unsqueeze(0)  # Add channel dimension
        #     consensus_mask = consensus_mask.unsqueeze(0)
        
        image = image.unsqueeze(0)  # Add channel dimension
        # consensus_mask = consensus_mask.unsqueeze(0)
        
        if self.return_2d_slices:
            image = image[:,:, image.shape[2] // 2] # Considering where most of the volume is concentrated 
            consensus_mask = consensus_mask[:,:, image.shape[2] // 2]
            
        if self.return_individual_masks:
            individual_masks = torch.tensor(individual_masks, dtype=torch.long)
            if self.return_2d_slices:
                individual_masks = individual_masks[:,:, individual_masks.shape[2] // 2, :]
            return image, consensus_mask, individual_masks #, sample_name
        else:
            return image, consensus_mask #, sample_name
        
# ---- LIDC Dataset Creation Functions Based on Abstract Class ----

class LIDCDataset(Dataset_Class):
    def __init__(self, image_path, mask_path, uq_map_path, prediction_path, semantic_mapping_path, **kwargs: dict):
        super().__init__(image_path, mask_path, uq_map_path, prediction_path, semantic_mapping_path, **kwargs)
  
        self.image_path = Path(image_path)
        self.uq_map_path = Path(uq_map_path)
        self.prediction_path = Path(prediction_path)
        self.semantic_mapping_path = semantic_mapping_path
        self.mask_path = Path(mask_path)
        
        # Extract kwargs with defaults if not provided
        self.task = kwargs.get('task', None)
        self.model_noise = kwargs.get('model_noise', None)
        self.uq_method = kwargs.get('uq_method', None)
        self.decomp = kwargs.get('decomp', None)
        self.spatial = kwargs.get('spatial', None)
        self.variation = kwargs.get('variation', None)
        self.data_noise = kwargs.get('data_noise', None)
        self.metadata = kwargs.get('metadata', False)
        self.render_2d = kwargs.get('render_2d', True)
        self.norm_input =  kwargs.get('norm_input', True)
        self.cons_thresh = kwargs.get('cons_thresh', True)
        self.render_ind_masks = kwargs.get('render_ind_masks', False)
        
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
        
    def _create_consensus_mask(self, sample_name):
        """Create consensus mask from multiple rater annotations"""
        gt_masks = []
        # Look for rater masks (0103_00_00, 0103_00_01, etc.)
        for rater_id in range(4):  # 4 raters (00, 01, 02, 03)
            mask_file = self.mask_path / f"{sample_name}_{rater_id:02d}.nii.gz"
            
            if mask_file.exists():
                mask_data = self._load_nifti(mask_file)
                if mask_data is not None:
                    gt_masks.append(mask_data)
                else:
                    print(f"Warning: Could not load {mask_file}")
            else:
                print(f"Warning: Rater mask not found: {mask_file}")
                
        if not gt_masks:
            raise ValueError(f"No valid rater masks found for sample {sample_name}")
            
        # Stack all rater masks and create consensus
        gt_stack = np.stack(gt_masks, axis=-1)  # Shape: (H, W, D, num_raters)
        # Create consensus based on threshold
        if self.cons_thresh == 1: #ValUES consensus
            # Any rater agrees (union)
            consensus_mask = np.any(gt_stack, axis=-1).astype(np.uint8)
        else:
            # At least N raters agree
            consensus_mask = (np.sum(gt_stack, axis=-1) >= self.cons_thresh).astype(np.uint8)
            
        if self.render_ind_masks:
            return consensus_mask, gt_stack
        else:
            return consensus_mask
        
    def __len__(self):
        return len(self.get_sample_names())
    
    def __getitem__(self, idx):
        
        if idx >= self.__len__():
            raise IndexError("Index out of bounds.")

        mask = self.get_mask(idx)
        if isinstance(mask, tuple):
            cons_mask, indiv_mask = mask 
        else:
            cons_mask = mask
            indiv_mask = torch.zeros_like(cons_mask)
                
        data = {
            'image': self.get_image(idx),
            'mask': self.get_mask(idx),
            'individual_masks' : indiv_mask,
            'uq_map': self.get_uq_map(idx),
            'prediction': self.get_prediction(idx),
            'sample_name': self.get_sample_name(idx),
        }
        return data
    
    def _load_nifti(self, file_path):
        """Load NIfTI file and return data"""
        try:
            img = nib.load(str(file_path))
            data = img.get_fdata()
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def get_image(self, idx):
        sample_name = self.get_sample_name(idx)
        input_file = self.image_path / f"{sample_name}.nii.gz"  # Load input image

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        image = self._load_nifti(input_file)
        if image is None:
            raise ValueError(f"Could not load input image: {input_file}")
        
        if self.norm_input:
            # For medical images, typically normalize to [0,1] based on data range
            image = normalize_min_max(image, None, None, clip=True)
        
        # Convert to tensors
        image = torch.tensor(image, dtype=torch.float32)
        image = image.unsqueeze(0)  # Add channel dimension
        
        if self.render_2d:
            return image[:,:, image.shape[2] // 2] # Considering where most of the volume is concentrated 
        return image
    
    def get_mask(self, idx):
        sample_name = self.get_sample_name(idx)
            
        # Load and create consensus mask
        if self.render_ind_masks:
            consensus_mask, individual_masks = self._create_consensus_mask(sample_name)
        else:
            consensus_mask = self._create_consensus_mask(sample_name)
                  
        # Convert to tensors
        consensus_mask = torch.tensor(consensus_mask, dtype=torch.long)
                   
        if self.render_2d:
            consensus_mask = consensus_mask[:,:, consensus_mask.shape[2] // 2]
            
        if self.render_ind_masks:
            individual_masks = torch.tensor(individual_masks, dtype=torch.long)
            if self.render_2d:
                individual_masks = individual_masks[:,:, individual_masks.shape[2] // 2, :]
            return consensus_mask, individual_masks 
        else:  
            return consensus_mask
    
    @lru_cache(maxsize=32)
    def get_uq_map(self, idx):
        """Load uncertainty maps"""
        sample_name = self.get_sample_name(idx)
        if self.metadata:
            fn = self._get_metadata_index(sample_name)  # Use metadata index to get correct position in uq_map array
        else:
            fn = idx  # Fallback: assume sample_names order matches array order
        
        map_type = f"{self.task}_noise_{self.model_noise}_{self.variation}_{self.data_noise}_{self.uq_method}_{self.decomp}"
        if self.spatial:
            map_type += f'_{self.spatial}'
        map_type += ".npy"
        map_file = self.uq_map_path .joinpath(map_type)
        return np.load(map_file)[fn]
    
    def get_prediction(self, idx, **kwargs):
        """Load Fg. Bg. 3D-UNet predictions"""
        sample_name = self.get_sample_name(idx)
        if self.metadata:
            fn = self._get_metadata_index(sample_name)  # Use metadata index to get correct position in uq_map array
        else:
            fn = idx  # Fallback: assume sample_names order matches array order
                
        preds_type = f"fgbg_noise_{self.model_noise}_{self.variation}_{self.data_noise}_{self.uq_method}.npy"
        preds_file_path = self.prediction_path.joinpath(preds_type)
        return np.load(preds_file_path)[fn]
    
    def get_sample_name(self, idx):
        """Return the sample name at the given index."""
        return self.get_sample_names()[idx]
             
    def get_sample_names(self):
        """Extract unique sample names from input directory"""
        if not self.image_path.exists():
            raise ValueError(f"Input directory does not exist: {self.image_path}")
            
        input_files = list(self.image_path.glob("*.nii.gz"))
        if not input_files:
            raise ValueError(f"No .nii.gz files found in {self.image_path}")
            
        # Extract base sample names (e.g., "0103_00" from "0103_00.nii.gz")
        self.sample_names = []
        for file in input_files:
            sample_name = file.stem.replace('.nii', '')  # Remove .nii from .nii.gz
            self.sample_names.append(sample_name)
            
        return sorted(list(set(self.sample_names)))
    
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
            'metadata': None  # TODO: Add metadata if available
        }
        return info_dictionary

def main():
    spatial = False
    main_folder_name = "UQ_maps" if not spatial else "UQ_spatial"
    base_path = Path('/fast/AG_Kainmueller/data/ValUES/')
    map_path = base_path
    
    extra_info = {
        'task' : 'fgbg',
        'variation' : 'malignancy',
        'model_noise' : 0,
        'data_noise': '1_00',
        'uq_method' : 'dropout',
        'decomp' : 'pu',
        'spatial' : None,
        'cons_thresh' : 2,
        'metadata' : True,
        'render_2d' : True,
        'render_ind_masks': False,
    }
    
    # Set up paths based on folder structure
    cycle = 'FirstCycle'
    folder = f"{extra_info['variation']}_fold0_seed123"
    placeholder = "Softmax"
    data_path = base_path.joinpath(f"{cycle}/{placeholder}/test_results/{folder}/") 
    
    if extra_info['data_noise'] == "0_00":
        data_dir = data_path / "id"
    else:
        data_dir = data_path / "ood"
        
    image_path = data_dir / "input"
    mask_path = data_dir / "gt_seg"
    prediction_path = map_path.joinpath('UQ_predictions')
    uq_map_path = map_path.joinpath(main_folder_name)
    
    data_loader = LIDCDataset(image_path, 
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
    
    # Check output and its dimensions
    data = next(iter(loader))
    
    print(data['image'].shape,
          data['mask'].shape, 
          data['uq_map'].shape, 
          data['prediction'].shape,
          data['sample_name'])
    
    # Assuming batch size B=1, squeeze to get rid of batch dimension
    image = data['image'].squeeze(0)  # Shape: C x H x W
    mask = data['mask'].squeeze(0)    # Shape: H x W
    uq_map = data['uq_map'].squeeze(0)  # Shape: H x W
    prediction = data['prediction'].squeeze(0)  # Shape: H x W
    sample_name = data['sample_name'][0]  # Assuming it's a list of strings

    # Convert image to H x W for visualization, e.g., use the first channel
    if image.ndim == 3:
        image_to_show = image[0]  # Shape: H x W
    else:
        image_to_show = image  # Already H x W

    # Create subplots
    fig, axs = plt.subplots(1, 4, figsize=(14, 5))
    titles = ['Input Image', 'Ground Truth', 'Prediction', 'UQ Map']
    overlays = [None, mask, prediction, uq_map]
    cmaps = [None, 'Purples', 'Purples', 'inferno']
    alphas = [1.0, 0.5, 0.5, 0.8]  # transparency for overlays

    for ax, title, overlay, cmap, alpha in zip(axs, titles, overlays, cmaps, alphas):
        ax.imshow(image_to_show, cmap='gray')
        if overlay is not None:
            ax.imshow(overlay, cmap=cmap, alpha=alpha)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    # Add sample name as the overall title
    fig.suptitle(f"Sample: {sample_name}", fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # Save the figure in the same directory as this script
    output_dir = Path(__file__).parent
    output_file = output_dir / 'sample_batch_overlay_plot.png'
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

    print(f"Overlay plot saved to {output_file}")
    
    
    
if __name__ == "__main__":
    main() 