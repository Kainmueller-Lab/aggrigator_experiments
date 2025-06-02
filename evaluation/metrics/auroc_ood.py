import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Any, Tuple, Optional
from sklearn.metrics import roc_curve, auc

from aggrigator.uncertainty_maps import UncertaintyMap

# ---- AUROC computation to assess an aggregator's ability to detect OoD images. ----

def compute_auroc_from_maps(
    uncertainty_maps: List[UncertaintyMap],
    gt_labels: np.ndarray,
    aggregation_method: Callable,
    params: Any,
    category: str
    ) -> float:
    """
    Compute AUROC to assess an aggregator's ability to detect OoD images, by defining:
    - True Positive Rate (TPR) as the proportion of correctly identified OoD images, 
    i.e., the fraction of OoD samples whose aggregated uncertainty score exceeds a given threshold. 
    - False Positive Rate (FPR) as the proportion of iD images incorrectly classified as OoD, 
    i.e., the fraction of iD samples whose aggregated uncertainty score surpasses the same threshold. 
     
    Parameters
    ----------
    uncertainty_maps : List[UncertaintyMap]
        List of uncertainty maps
    gt_labels : np.ndarray
        Ground truth labels (0 for in-distribution, 1 for out-of-distribution)
    aggregation_method : Callable
        Function to aggregate uncertainty values
    params : Any
        Parameters for aggregation method
    category : str
        Category of aggregation method (e.g., 'Threshold', 'Spatial')
    
    Returns
    -------
    float
        AUROC value by computing TPRs and FPRs at differet thresholds via sklearn library
    """
    # Apply aggregation method
    uncertainty_values = np.array([
        aggregation_method(umap, params) for umap in uncertainty_maps
    ])
    
    # Handle threshold methods
    if category == 'Threshold': 
        uncertainty_values = np.nan_to_num(uncertainty_values, nan=0)
        mask = (uncertainty_values == -1) | (uncertainty_values == 0)
        uncertainty_values[mask] = 0

    # Calculate AUROC
    fpr, tpr, _ = roc_curve(gt_labels, uncertainty_values)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def evaluate_aggregation_strategy(
    cached_maps: Dict,
    uq_methods: List[str],
    aggr_name: str,
    aggr_method: Callable,
    param: Any,
    category: str
    ) -> Dict:
    """
    Evaluate an aggregation strategy across multiple UQ methods.
    
    Parameters
    ----------
    cached_maps : Dict
        Preloaded uncertainty maps for each UQ method
    uq_methods : List[str]
        List of UQ methods to evaluate
    aggr_name : str
        Name of aggregation method
    aggr_method : Callable
        Aggregation method function
    param : Any
        Parameters for aggregation method
    category : str
        Category of aggregation method
    
    Returns
    -------
    Dict
        Results with AUROC mean and standard deviation
    """
    # Compute AUROC for each UQ method using preloaded maps
    auroc_values = np.zeros(len(uq_methods))
    
    for idx, uq_method in enumerate(uq_methods):
        uncertainty_maps = cached_maps[uq_method]['maps']
        gt_labels = cached_maps[uq_method]['gt_labels']
        auroc_values[idx] = compute_auroc_from_maps(
            uncertainty_maps, gt_labels, aggr_method, param, category
        )
    
    # Return results
    return {
        'Aggregator': aggr_name,
        'AUROC': np.mean(auroc_values),
        'AUROC_std': np.std(auroc_values),
    }


def evaluate_all_strategies(
    cached_maps: Dict,
    strategies: Dict,
    noise_level: str,
    decomp: str
    ) -> pd.DataFrame:
    """
    Evaluate all aggregation strategies for a given noise level.
    
    Parameters
    ----------
    cached_maps : Dict
        Preloaded uncertainty maps
    strategies : Dict
        Dictionary of aggregation strategies
    noise_level : str
        Current noise level
    
    Returns
    -------
    pd.DataFrame
        DataFrame with AUROC results for all strategies
    """
    uq_methods = ['softmax', 'ensemble', 'dropout', 'tta'] if decomp == 'pu' else ['ensemble', 'dropout', 'tta']
    auroc_data = []
    
    # Process each aggregation strategy
    for category, methods in strategies.items():
        for aggr_name, (aggr_method, param) in methods.items():
            try:
                print(f"----Processing aggregator function: {aggr_name}, in {category} category----")
                # Evaluate the strategy
                result = evaluate_aggregation_strategy(
                    cached_maps, uq_methods, aggr_name, aggr_method, param, category
                )
                # Add noise level
                result['Noise_Level'] = noise_level
                # Store results
                auroc_data.append(result)
            
            except Exception as e:
                print(f"Error processing method {aggr_method} for noise level {noise_level}: {e}")
                continue
    
    # Convert to DataFrame and sort by AUROC
    df = pd.DataFrame(auroc_data)
    df = df.sort_values('AUROC', ascending=False).reset_index(drop=True)
    return df