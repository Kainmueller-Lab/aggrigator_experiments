import sys
from pathlib import Path

current_dir = Path.cwd()
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))
utils_dir = current_dir.parent / 'AggroUQ' / 'src'
sys.path.append(str(utils_dir))

from aggrigator.methods import AggregationMethods as am

CLASS_NAMES_ARCTIQUE = {
    "epithelial-cell": 1,
    "plasma-cell": 2,
    "lymphocite": 3, 
    "eosinophil": 4,
    "fibroblast": 5
}

STRATEGIES = {
        'Baseline': {
                'Mean': (am.mean, None), 
                # 'Sum': (am.sum, None),
                # 'Max': (am.max, None),
            },
        # 'Spatial': {
        #         'Morans' : (am.morans_I, None),
        #                 },
        'Threshold':{
                'Threshold 0.3': (am.above_threshold_mean, 0.3),
                # 'Threshold 0.4': (am.above_threshold_mean, 0.4),
                'Threshold 0.7': (am.above_threshold_mean, 0.7),
            },
        'Quantile':{
        #     'Quantile 0.6': (am.above_quantile_mean, 0.6),
            'Quantile 0.7': (am.above_quantile_mean, 0.7),
        #     'Quantile 0.9': (am.above_quantile_mean, 0.6),
            },
        'Patch':{
                # 'Patch 10': (am.patch_aggregation, 10), 
                # 'Patch 20': (am.patch_aggregation, 20),
                'Patch 40': (am.patch_aggregation, 40),
                # 'Patch 80': (am.patch_aggregation, 80),
                # 'Patch 150': (am.patch_aggregation, 150),
                # 'Patch 200': (am.patch_aggregation, 200),
                # 'Patch 50': (am.patch_aggregation, 20),
            },
    }

CLASS_STRATEGIES = {
    'Class-based': {
        "equally-weighted average": (am.class_mean_w_equal_weights, None),
        "imbalance-weighted average": (am.class_mean_weighted_by_occurrence, None),
    }
}
