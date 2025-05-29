import sys
import seaborn as sns
import matplotlib.cm as cm 
from pathlib import Path
from aggrigator.methods import AggregationMethods as am

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
                'Equally-w. class avg.' : (am.class_mean_w_equal_weights, None),
                'Imbalance-w. class avg.': (am.class_mean_weighted_by_occurrence, None),
                        },
        'Baseline': {
                'Mean': (am.mean, None), 
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

BACKGROUND_FREE_STRATEGIES = {
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

tab20 = cm.get_cmap('tab20', 20)  # Get the tab20 colormap with 20 colors
COLORS = [tab20(i) for i in range(7)]
