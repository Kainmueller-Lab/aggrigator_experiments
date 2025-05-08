import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from functools import lru_cache

from aggrigator.uncertainty_maps import UncertaintyMap
from data_utils import load_unc_maps, load_dataset, rescale_maps

# ---- Utilities for loading and manipulating uncertainty maps ----

def preload_uncertainty_maps(
    uq_path: Path, 
    metadata_path: Path, 
    gt_list: List[np.ndarray], 
    task: str, 
    model_noise: int, 
    variation: str, 
    data_noise: str
) -> Dict[str, Dict]:
    """
    Preload all uncertainty maps for a given noise level, their metadata indices
    and iD and OoD image targets as either 0 or 1 to then calculate the aggregators AUROC score.
    
    Parameters
    ----------
    uq_path : Path
        Path to uncertainty maps
    metadata_path : Path
        Path to metadata files
    gt_list : List[np.ndarray]
        List of ground truth arrays
    task : str
        Task type ('instance' or 'semantic')
    model_noise : int
        Model noise level
    variation : str
        Variation type
    data_noise : str
        Data noise level
    
    Returns
    -------
    Dict[str, Dict]
        Dictionary containing loaded maps and metadata for each UQ method
    """
    uq_methods = ['softmax', 'ensemble', 'dropout', 'tta']
    idx_task = 2 if task == 'instance' else 1
    gt_array = np.array(gt_list)[..., idx_task]
    
    # Dictionary to store loaded maps for each UQ method
    cached_maps = {}
    
    for uq_method in uq_methods:
        # Load zero-risk and noisy uncertainty maps
        uq_maps_zr, metadata_file_zr = load_unc_maps(
            uq_path, task, model_noise, variation, '0_00', 
            uq_method, 'pu', False, metadata_path
        )
        uq_maps_r, metadata_file_r = load_unc_maps(
            uq_path, task, model_noise, variation, data_noise, 
            uq_method, 'pu', False, metadata_path
        )
        
        # Normalize when needed
        uq_maps_zr = rescale_maps(uq_maps_zr, uq_method, task)
        uq_maps_r = rescale_maps(uq_maps_r, uq_method, task)
        
        # Concatenate maps
        uq_maps = np.concatenate((uq_maps_zr, uq_maps_r), axis=0)
        
        # Setup context masks
        context_gt = np.concatenate([gt_array, gt_array], axis=0)
        
        # Create UncertaintyMap objects
        uncertainty_maps = [
            UncertaintyMap(array=array, mask=gt, name=None) 
            for (array, gt) in zip(uq_maps, context_gt)
        ]
        
        # Define iD and OoD targets
        gt_labels_0 = np.zeros((len(uq_maps_zr)))
        gt_labels_1 = np.ones((len(uq_maps_r)))
        gt_labels = np.concatenate((gt_labels_0, gt_labels_1), axis=0)
        
        # Store in cache
        cached_maps[uq_method] = {
            'maps': uncertainty_maps,
            'gt_labels': gt_labels,
            'metadata': [metadata_file_zr, metadata_file_r]
        }
    
    return cached_maps


def get_aggregator_method_params(strategies_dict: Dict) -> Dict[str, List[Tuple]]:
    """
    Organize aggregation methods by category.
    
    Parameters
    ----------
    strategies_dict : Dict
        Dictionary of strategies by category
        
    Returns
    -------
    Dict[str, List[Tuple]]
        Methods organized by category with parameters
    """
    organized_methods = {}
    
    for category, methods in strategies_dict.items():
        if category not in organized_methods:
            organized_methods[category] = []
            
        for method_name, (method_func, params) in methods.items():
            organized_methods[category].append((method_name, method_func, params))
    
    return organized_methods