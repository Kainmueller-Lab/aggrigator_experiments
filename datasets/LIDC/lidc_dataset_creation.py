import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import nibabel as nib
import glob
from typing import Optional, Tuple

# ---- LIDC_IDRI Dataset Creation Functions ----

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
        self._get_sample_names()
        
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
            
        self.sample_names = sorted(list(set(self.sample_names)))
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