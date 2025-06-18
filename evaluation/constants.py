import sys
import numpy as np
import seaborn as sns
import matplotlib.cm as cm 
from pathlib import Path
from aggrigator.methods import AggregationMethods as am

def class_mean(unc_map, param):
    assert unc_map.mask_provided, f"Mask not provided for uncertainty map {unc_map.name}"
    assert param in unc_map.class_indices, f"Invalid class label {param} for uncertainty map {unc_map.name}"
    return np.sum(unc_map.array[unc_map.mask == param], dtype=np.float64) / unc_map.class_volumes[param]
    
def get_id_mask(mask, id):
    return np.where(mask==id, 1, 0)

def class_mean_w_custom_weights(unc_map, param): # param = weights: A dict of weights for each class you want to include.
    """
    Compute the weighted average of class means, allowing for custom weights.
    Parameters:
    - unc_map: An object containing class indices and a method to compute class means.
    - param (dict, optional): A dictionary specifying custom weights for each class.
    Returns:
    - Weighted average of class means.
    """
    assert unc_map.mask_provided, f"Mask not provided for uncertainty map {unc_map.name}"
    weights = param
    class_ids = list(weights.keys())
    # Compute class means
    class_means = {class_id: class_mean(unc_map, class_id)
                    for class_id in class_ids}
    # Ensure provided weights sum to 1
    weight_sum = sum(weights.values())
    assert abs(weight_sum - 1.0) < 1e-6, "Weights must sum to 1."
    return sum(class_means[id] * weights[id] for id in class_ids)

def class_mean_w_equal_weights(unc_map, param=False, ignore_index=255):
    # NOTE: We exclude BG class 0 if include_background is False
    classes = [class_id for class_id in unc_map.class_indices if not (class_id == ignore_index and param)]
    # Use equal weights for all classes
    weights = {id: 1 / len(classes) for id in classes}
    return class_mean_w_custom_weights(unc_map, weights)

def class_mean_weighted_by_occurrence(unc_map, param=False, ignore_index=255):
    # NOTE: We exclude BG class 0 if include_background is False
    classes = [class_id for class_id in unc_map.class_indices if not (class_id == ignore_index and param)]
    # Count class pixels 
    class_pixel_counts = {class_id: get_id_mask(unc_map.mask, class_id).sum()
                            for class_id in classes}
    fg_pixel_count = np.sum(list(class_pixel_counts.values()))
    # Use weights proportional to the number of pixels in each class
    weights = {id: class_pixel_counts[id] / fg_pixel_count for id in classes}
    return class_mean_w_custom_weights(unc_map, weights)

AURC_DISPLAY_SCALE = 1000

CLASS_NAMES_ARCTIQUE = {
    "epithelial-cell": 1,
    "plasma-cell": 2,
    "lymphocite": 3, 
    "eosinophil": 4,
    "fibroblast": 5
}

CLASS_NAMES_LIZARD = {
    "epithelial-cell": 1,
    "plasma-cell": 2,
    "lymphocite": 3, 
    "eosinophil": 4,
    "connective-tissue-cell": 5,
    "neutrophil": 6
}

NOISE_LEVELS_ARCTIQUE = ["0_25", "0_50", "0_75", "1_00"]
NOISE_LEVELS = ["1_00"]

AUROC_STRATEGIES = {
        'Context-aware': {
                'Equally-w. class avg.' : (class_mean_w_equal_weights, True),
                'Imbalance-w. class avg.': (class_mean_weighted_by_occurrence, True),
                        },
        'Baseline': {
                'Mean': (am.mean, None), 
            },
        'Threshold':{
                # 'Threshold 0.2': (am.above_threshold_mean, 0.2),
                'Threshold 0.3': (am.above_threshold_mean, 0.3),
                'Threshold 0.4': (am.above_threshold_mean, 0.4),
                'Threshold 0.5': (am.above_threshold_mean, 0.5),
                # 'Threshold 0.6': (am.above_threshold_mean, 0.6),
                # 'Threshold 0.7': (am.above_threshold_mean, 0.7),
                # 'Threshold 0.8': (am.above_threshold_mean, 0.8),
                # 'Threshold 0.9': (am.above_threshold_mean, 0.9),
            },
        'Quantile':{
                'Quantile 0.6': (am.above_quantile_mean, 0.6),
                'Quantile 0.75': (am.above_quantile_mean, 0.75),
                'Quantile 0.9': (am.above_quantile_mean, 0.9),
                'Quantile fg. ratio' : (am.above_quantile_mean_fg_ratio, None)
            },
        'Patch':{
                'Patch 10': (am.patch_aggregation, 10), 
                'Patch 20': (am.patch_aggregation, 20),
                'Patch 50': (am.patch_aggregation, 50),
            }
    }

BACKGROUND_FREE_STRATEGIES = {
    'Context-aware': {
        "equally-weighted average": (am.class_mean_w_equal_weights, None),
        "imbalance-weighted average": (am.class_mean_weighted_by_occurrence, None),
    }
}

BARPLOTS_COLORS = {
    'Baseline': "#A31212",
    'Context-aware' : "#BDB76B",
    'Threshold': sns.light_palette("blue", n_colors=6)[1],
    'Quantile': sns.light_palette("blue", n_colors=6)[2],
    'Patch': sns.light_palette("blue", n_colors=6)[3],
}

tab20 = cm.get_cmap('tab20', 20)  # Get the tab20 colormap with 20 colors
COLORS = [tab20(i) for i in range(7)]
