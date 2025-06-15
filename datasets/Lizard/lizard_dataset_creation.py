import torch
import numpy as np 
import mahotas as mh 
import lmdb
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import json

from torch.utils.data import Dataset, DataLoader
from datasets.dataset import Dataset_Class
from pathlib import Path
from functools import lru_cache

# ---- Lizard Config. Functions ----

# cell IDS from https://github.com/digitalpathologybern/hover_next_train/blob/main/src/constants.py whose order is modified according to those of Arctique 
mapping_dict ={
        "0": "background",
        "1": "Epithelial",
        "2": "Plasma",
        "3": "Lymphocyte",
        "4": "Eosinophil",
        "5": "Connective tissue",
        "6": "Neutrophil",
}

# ---- Lizard Dataset Creation Functions ----

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

class LizardDataset(Dataset):
    def __init__(self, root_dir, mode, norm=True): # mode is either train, val or test 
        self.mode = mode
        self.root_dir = root_dir
        self.norm = norm
        
        if mode == 'train':
            self.images_file = 'images_train.npy'
            self.labels_file = 'labels_train.npy'  
        elif mode == 'val':
            self.images_file = 'images_val.npy'
            self.labels_file = 'labels_val.npy' 
        elif mode == 'test': 
            self.images_file = 'images_test.npy'
            self.labels_file = 'labels_test.npy' 
        
        self.images = np.load(Path(root_dir).joinpath(self.images_file))
        self.labels = np.load(Path(root_dir).joinpath(self.labels_file))
        
        #match fibroblasts to the connective tissue 
        self.class_mapping = {0: 0, 1: 6, 2: 1, 3: 3, 4: 2, 5: 4, 6: 5} 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):        
        image_tmp = self.images[idx].astype(np.float32)
        if self.norm:
            image_tmp = normalize_min_max(image_tmp, 0, 255)
        image_tmp = torch.tensor(image_tmp, dtype=torch.float32) 
        image =  image_tmp.permute(2, 0, 1) # CxHxW
        
        if self.labels is not None:
            lab_tmp = self.labels[idx].astype(np.float32)
            labinst_tmp, labsem_tmp, lab3c_tmp = lab_tmp[...,0], lab_tmp[...,1], lab_tmp[...,2]
        
        # if self.mode == 'val' or self.mode == 'test':
        labsem_tmp = self.rearrange_class(labsem_tmp)
        label = np.stack((labinst_tmp, labsem_tmp, lab3c_tmp), axis=-1)
        label = torch.tensor(label, dtype=torch.int64)
        
        return image, label
    
    def rearrange_class(self, sem_label):
        vectorized_mapping = np.vectorize(lambda x: self.class_mapping.get(x, x))
        return vectorized_mapping(sem_label) 
    
# ---- Lizard Dataset Creation Functions Based on Abstract Class ----

class LizardDataset(Dataset_Class):
    def __init__(self, image_path, mask_path, uq_map_path, prediction_path, semantic_mapping_path, **kwargs: dict):
        super().__init__(image_path, mask_path, uq_map_path, prediction_path, semantic_mapping_path, **kwargs)
        """Initializes the LMDBDataset.

        Args:
            path (str): Path to the LMDB file.
            include_sample_names (list, optional): List of sample names to include. Default is None, meaning all samples are included.
        """
        self.lmdb_path = image_path

        # Open the LMDB environment
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        
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
        self.split = kwargs.get('split', ['test'])
        
        assert any(s in self.split for s in {"train", "test", "valid"})
        
        if self.split_path:
            self.get_split()
        else:
            self.include_sample_names = None
            self.include_tile_names = None

        # load metadata
        self.metadata = self._load_metadata()
        self.size = self.metadata["tile_size"]
        self.microns_per_pixel = self.metadata["microns_per_pixel"]
        self.class_mapping_dict = self.metadata["class_mapping_dict"]
        self.n_classes = self.metadata["n_classes"]
        
        # rearrange mapping for comparison with Arctique 
        self.class_mapping = {0: 0, 1: 6, 2: 1, 3: 3, 4: 2, 5: 4, 6: 5} 

        # Filter the keys based on sample names
        self._filter_lmdb_indices()
    
    def get_split(self):
        # load split .csv
        split_df = pd.read_csv(self.split_path)
        # enforce column names
        assert {"sample_name", "train_test_val_split"} == set(split_df.columns)
        # enfore split names
        assert {"train", "test", "valid"} == set(split_df["train_test_val_split"].unique())
        # load intended split
        for split in self.split:
            include_fovs = split_df[split_df["train_test_val_split"] == split]["sample_name"].tolist()
            if "tile" in str(include_fovs[0]).lower():
                self.include_tile_names = include_fovs
                self.include_sample_names = None
            else:
                self.include_tile_names = None
                self.include_sample_names = include_fovs

    def _load_metadata(self):
        """Loads metadata from the LMDB file."""
        with self.env.begin(write=False) as txn:
            meta_bytes = txn.get(b"__global_metadata__")
            if meta_bytes is None:
                raise ValueError("No global metadata found in the LMDB file.")
            metadata = pickle.loads(meta_bytes)
        return metadata

    def _filter_lmdb_indices(self):
        """Filters LMDB keys based on the include_sample_names list."""
        with self.env.begin(write=False) as txn:
            # Retrieve all keys and associated sample names
            include_keys = []

            cursor = txn.cursor()
            for key, value in cursor:
                if key.startswith(b"tile-"):

                    data = pickle.loads(value)
                    if self.include_tile_names is not None:
                        if data["tile_index"] in self.include_tile_names:
                            include_keys.append(key)
                            continue

                    if self.include_sample_names is not None:
                        if data["sample_name"] in self.include_sample_names:
                            include_keys.append(key)
                            continue

                    else:  # Include all keys if include_sample_names is None and include_tile_names is None
                        include_keys.append(key)

        self.lmdb_key_list = include_keys

    def _pad_image_data(self, data, key='image'):
        """Pads the image data to ensure all images are of the same size.

        Args:
            data (array): The data dictionary values for one key
            keys (list): List of keys to pad.

        Returns:
            data (array): The padded data values dictionary.
        """
        target_size = self.metadata["tile_size"]
        content_mask = np.zeros((target_size, target_size), dtype=np.uint8)
        if isinstance(data, np.ndarray):
            h, w = data.shape[:2]
            pad_h = target_size - h
            pad_w = target_size - w
            if pad_h > 0 or pad_w > 0:
                if len(data.shape) == 2:  # Grayscale image
                    data = np.pad(
                        data, ((0, pad_h), (0, pad_w)), 
                        mode='constant', constant_values=0)
                elif len(data.shape) == 3:  # Color image
                    data = np.pad(
                        data, ((0, pad_h), (0, pad_w), (0, 0)), 
                        mode='constant', constant_values=0)
                content_mask[:h, :w] = 1
        else:
            raise ValueError(f"Unsupported data type for key {key}: {type(data)}")
        return data, content_mask
    
    def _image_to_tensor(self, data, keys=['image', 'mask']):
        """Converts image data to PyTorch tensors.

        Args:
            data (dict): The data dictionary containing images and masks.
            keys (list): List of keys to convert.

        Returns:
            dict: The data dictionary with images and masks converted to tensors.
        """
        for k, v in data.items():
            for key in keys:
                if key in k:
                    if isinstance(v, np.ndarray):
                        data[k] = torch.from_numpy(v).float()
                    else:
                        raise ValueError(f"Unsupported data type for key {key}: {type(v[key])}")
        data['image'] = data['image'].permute(2, 0, 1)  # HWC to CHW
        return data

    def __len__(self):
        """Returns the number of filtered tiles in the dataset."""
        return len(self.lmdb_key_list)

    def __getitem__(self, idx):
        """Retrieves a tile from the LMDB file based on the given index.

        Args:
            idx (int): Index of the tile to retrieve.

        Returns
        -------
            dict: The data associated with the tile.
        """
        if idx < 0 or idx >= len(self.lmdb_key_list):
            raise IndexError("Index out of range.")

        data = {
            'image': self.get_image(idx),
            'mask': self.get_mask(idx),
            'uq_map': self.get_uq_map(idx),
            'prediction': self.get_prediction(idx),
            'sample_name': self.get_sample_name(idx),
        }

        return data
    
    def get_image(self, idx):
        """Get image for a specific index."""
        key = self.lmdb_key_list[idx]
        with self.env.begin(write=False) as txn:
            value = txn.get(key)
            if value is None:
                raise ValueError(f"No data found for key {key}")
            data = pickle.loads(value)
        
        if "image" in data and data["image"] is not None:
            if data["image"].max() > 1:  # Normalize image if it is not already normalized
                data["image"] = data["image"] / data["image"].max()
        
        image, img_content_mask = self._pad_image_data(data.get('image'), 'image')
        image = torch.tensor(image, dtype=torch.float32) 
        return image.permute(2, 0, 1) # CxHxW  
        
    def get_mask(self, idx):
        """Get mask for a specific index."""
        key = self.lmdb_key_list[idx]
        with self.env.begin(write=False) as txn:
            value = txn.get(key)
            if value is None:
                raise ValueError(f"No data found for key {key}")
            data = pickle.loads(value)
            # print(data.keys())
        
        sem_mask, sem_content_mask = self._pad_image_data(data.get('semantic_mask'), 'semantic_mask')
        sem_mask = self.rearrange_class(sem_mask)
        inst_mask, inst_content_mask = self._pad_image_data(data.get('instance_mask'), 'instance_mask')
        three_mask = inst_to_3c(inst_mask, False)
        
        # Check both 'mask' and 'masks' keys
        if self.task.startswith('semantic'):
            return torch.from_numpy(sem_mask) if isinstance(sem_mask, np.ndarray) else sem_mask
        elif self.task.startswith('instance'):
            mask = np.stack((inst_mask, three_mask), axis=-1)
            return torch.from_numpy(mask) if isinstance(mask, np.ndarray) else mask
        
        mask = np.stack((inst_mask, sem_mask, three_mask), axis=-1)
        return torch.tensor(mask, dtype=torch.long)
    
    @lru_cache(maxsize=32)
    def get_uq_map(self, idx):
        """Load uncertainty maps"""
        return torch.zeros_like(self.get_mask(idx))
    
    def get_prediction(self, idx, **kwargs):
        """Load panoptic model predictions"""
        return torch.zeros_like(self.get_mask(idx))
    
    def close(self):
        """Closes the LMDB environment."""
        if self.env is not None:
            self.env.close()
            self.env = None
    
    def get_sample_name(self, idx):
        """Return the sample name at the given index."""
        key = self.lmdb_key_list[idx]
        with self.env.begin(write=False) as txn:
            value = txn.get(key)
            if value is not None:
                data = pickle.loads(value)
                return f'{data["sample_name"]}_{data["tile_index"]}'
        return None
             
    def get_sample_names(self):
        return self.get_sample_names_in_dataset()
    
    def get_sample_names_in_dataset(self):
        """Get all unique sample names in the current filtered dataset."""
        sample_names = set()
        for key in self.lmdb_key_list:
            with self.env.begin(write=False) as txn:
                value = txn.get(key)
                if value is not None:
                    data = pickle.loads(value)
                    sample_names.add(f'{data["sample_name"]}_{data["tile_index"]}')
        return list(sample_names)
    
    def get_semantic_mapping(self):
        return mapping_dict
    
    def rearrange_class(self, sem_label):
        vectorized_mapping = np.vectorize(lambda x: self.class_mapping.get(x, x))
        return vectorized_mapping(sem_label) 
    
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
    spatial = False
    main_folder_name = "UQ_maps" if not spatial else "UQ_spatial"
    lmdb_path = '/fast/AG_Kainmueller/data/Lizard/lizard_lmdb/'
    # base_path = Path('/fast/AG_Kainmueller/synth_unc_models/data/v1-0-variations/variations/')
    extra_info = {
        'task' : 'instance',
        'variation' : 'glas',
        'model_noise' : 0,
        'data_noise': '0_00',
        'uq_method' : 'dropout',
        'decomp' : 'pu',
        'spatial' : None,
        'metadata' : True,
        'split_path' : None,
        'split' : ['test']
    }
    
    csv_path = Path(lmdb_path).parent.joinpath(f"splits/domain_shift_splits/lizard_domaingen_{extra_info['variation']}_test_split.csv")
    extra_info['split_path'] = csv_path
    
    # image_path = base_path.joinpath(extra_info['variation'], extra_info['data_noise'], 'images')
    # mask_path = base_path.joinpath(extra_info['variation'], extra_info['data_noise'], 'masks')
    # prediction_path = map_path.joinpath('UQ_predictions')
    # uq_map_path = map_path.joinpath(main_folder_name)
    
    data_loader = LizardDataset(lmdb_path, 
                                lmdb_path, 
                                lmdb_path, 
                                lmdb_path, 
                                'abc',
                                **extra_info)
    print(data_loader.get_semantic_mapping())
    print(data_loader.__len__())
    
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
        2: [56, 153, 250],        # Plasma Cells - light blue
        3: [255, 255, 0],         # Lymphocytes - yellow
        4: [255, 105, 180],       # Eosinophils - reddish pink
        5: [0, 255, 0],           # Fibroblasts - green
        6: [255, 255, 255],       # Neutrophils - white
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
        # prediction = data['prediction'].squeeze(0).cpu().numpy()  # (H, W)
        label_colors = label_colors_sem        
    else:
        mask = data['mask'][...,1].squeeze(0).cpu().numpy() # (H, W) 
        # prediction = data['prediction'][...,1].squeeze(0).cpu().numpy()  # (H, W)
        label_colors = label_colors_inst
        
    # uq_map = data['uq_map'].squeeze(0)
    sample_name = data['sample_name'][0]

    # Generate colored overlays
    mask_rgb = label_to_rgb(mask, label_colors)
    # pred_rgb = label_to_rgb(prediction, label_colors)

    # Create subplots
    fig, axs = plt.subplots(1, 4, figsize=(16, 5))
    titles = ['Input Image', 'Ground Truth', 'Prediction', 'UQ Map']
    overlays = [None, mask_rgb, None, None]#, mask_rgb, pred_rgb, uq_map.cpu().numpy()]
    alphas = [1.0, 0.6, 0.6, 0.8]

    for ax, title, overlay, alpha in zip(axs, titles, overlays, alphas):
        if title == 'Input Image':
            ax.imshow(image.permute(1, 2, 0).cpu().numpy())  # RGB input, normalized
        else:
            ax.imshow(image[2], cmap='gray')
            if title in ['Ground Truth']:# ,'Prediction']:
                ax.imshow(overlay, alpha=alpha)
            # elif title == 'UQ Map':
            #     ax.imshow(overlay, cmap='inferno', alpha=alpha)
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

    