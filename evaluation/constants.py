import sys
import seaborn as sns
from pathlib import Path
from aggrigator.methods import AggregationMethods as am

CLASS_NAMES_ARCTIQUE = {
    "epithelial-cell": 1,
    "plasma-cell": 2,
    "lymphocite": 3, 
    "eosinophil": 4,
    "fibroblast": 5
}

NOISE_LEVELS = ["0_25", "0_50", "0_75", "1_00"]

AUROC_STRATEGIES = {
        'Baseline': {
                'Mean': (am.mean, None), 
            },
        'Context-aware': {
                'Equally-w. class avg.' : (am.class_mean_w_equal_weights, None),
                'Imbalance-w. class avg.': (am.class_mean_weighted_by_occurrence, None),
                        },
        'Threshold':{
                'Threshold 0.3': (am.above_threshold_mean, 0.3),
                'Threshold 0.4': (am.above_threshold_mean, 0.4),
                'Threshold 0.5': (am.above_threshold_mean, 0.5),
            },
        'Quantile':{
                'Quantile 0.6': (am.above_quantile_mean, 0.6),
            'Quantile 0.75': (am.above_quantile_mean, 0.75),
            'Quantile 0.9': (am.above_quantile_mean, 0.9),
            },
        'Patch':{
                'Patch 10': (am.patch_aggregation, 10), 
                'Patch 20': (am.patch_aggregation, 20),
                'Patch 50': (am.patch_aggregation, 50),
            }
    }

CLASS_STRATEGIES = {
    'Class-based': {
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
