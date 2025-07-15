import os
import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Any, Tuple, Optional
from sklearn.metrics import roc_curve, auc

from aggrigator.uncertainty_maps import UncertaintyMap

# ---- AUROC computation to assess an aggregator's ability to detect OoD images. ----

# --- Old Helper Functions ---

def compute_auroc_from_maps(
    uncertainty_maps: List[UncertaintyMap],
    gt_labels: np.ndarray,
    aggregation_method: Callable,
    params: Any,
    category: str,
    ignore_index: int
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
    
    # Apply aggregation method based on category
    if category == 'Context-aware':
        uncertainty_values = np.array([
            aggregation_method(umap, params, ignore_index) for umap in uncertainty_maps
        ])
    else:
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
    # uq_methods: List[str],
    aggr_name: str,
    aggr_method: Callable,
    param: Any,
    category: str,
    ignore_index: int
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
    auroc_values = np.zeros(len(list(cached_maps.keys())))
    
    for idx, uq_method in enumerate(list(cached_maps.keys())):
        uncertainty_maps = cached_maps[uq_method]['maps']
        gt_labels = cached_maps[uq_method]['gt_labels']
        auroc_values[idx] = compute_auroc_from_maps(
            uncertainty_maps, gt_labels, aggr_method, param, category, ignore_index
        )
    
    # Return results
    return {
        'Aggregator': aggr_name,
        'AUROC': np.mean(auroc_values),
        'AUROC_std': np.std(auroc_values),
    }


# --- New Helper Functions ---

def _evaluate_standard_strategies(
    cached_maps: Dict,
    strategies: Dict,
    noise_level: str,
    ignore_index: int
) -> List[Dict]:
    """
    Evaluates all standard pixel-based aggregation strategies.
    
    Returns:
        A list of result dictionaries.
    """
    auroc_data = []
    for category, methods in strategies.items():
        for aggr_name, (aggr_method, param) in methods.items():
            try:
                # print(f"----Processing aggregator function: {aggr_name}, in {category} category----")
                result = evaluate_aggregation_strategy(
                    cached_maps, aggr_name, aggr_method, param, category, ignore_index
                )
                result['Noise_Level'] = noise_level
                auroc_data.append(result)
            except Exception as e:
                print(f"Error processing method {aggr_method} for noise level {noise_level}: {e}")
                continue
    return auroc_data


def _evaluate_gmm_strategy(
    cached_maps: Dict,
    noise_level: str,
    dataset_name: str,
    task: str,
    variation: str,
    decomp: str
) -> Optional[Dict]:
    """
    Loads, aligns, and evaluates the pre-computed GMM score.
    
    Returns:
        A single result dictionary for the GMM score, or None if it cannot be computed.
    """
    first_uq_method = next(iter(cached_maps.keys()))
    if 'sample_names' not in cached_maps[first_uq_method]:
        return None

    # Construct the GMM scores filename and path
    scores_filename = f"{task}_{dataset_name}_{variation}_{decomp}_scores.csv"
    scores_filepath = os.path.join(os.getcwd(), "spatial", "results", scores_filename)
    
    if not os.path.exists(scores_filepath):
        print("Warning: GMM scores file not found, skipping GMM AUROC calculation.")
        return None

    print("----Processing aggregator function: GMM Normalized Score, in Spatial category----")
    
    # Load and prepare GMM scores with a multi-column key
    gmm_scores_df = pd.read_csv(scores_filepath)
    gmm_scores_df.rename(columns={'Unnamed: 0': 'uq_map_name', 'is_ood': 'gt_label'}, inplace=True)
    gmm_scores_df = gmm_scores_df.astype({'uq_map_name': str, 'gt_label': int})
    gmm_scores_to_merge = gmm_scores_df[['uq_map_name', 'gt_label', 'ood_score_normalized']]

    # Create alignment dataframe from cached data
    alignment_df = pd.DataFrame({
        'uq_map_name': cached_maps[first_uq_method]['sample_names'],
        'gt_label': cached_maps[first_uq_method]['gt_labels']
    }).astype({'uq_map_name': str, 'gt_label': int})

    # Perform an explicit merge on the composite key
    final_scores = pd.merge(
        alignment_df,
        gmm_scores_to_merge,
        on=['uq_map_name', 'gt_label'],
        how='left'
    )
    final_scores.rename(columns={'ood_score_normalized': 'gmm_score'}, inplace=True)
    final_scores.dropna(subset=['gmm_score'], inplace=True)

    if final_scores.empty:
        return None

    # Compute AUROC
    fpr, tpr, _ = roc_curve(final_scores['gt_label'], final_scores['gmm_score'])
    roc_auc = auc(fpr, tpr)
    
    return {
        'Aggregator': 'GMM Normalized',
        'AUROC': roc_auc,
        'AUROC_std': 0.0,
        'Noise_Level': noise_level,
    }


# --- Main Function ---

def evaluate_all_strategies(
    cached_maps: Dict,
    strategies: Dict,
    noise_level: str,
    ignore_index: int, 
    dataset_name: str,
    task: str,
    variation: str,
    decomp: str
) -> pd.DataFrame:
    """
    Evaluates all pixel-based and spatial aggregation strategies for a given noise level.
    """
    if not cached_maps:
        print(f"Warning: cached_maps dictionary is empty for noise level {noise_level}. No strategies will be evaluated.")
        return pd.DataFrame()
    
    # Step 1: Evaluate all standard pixel-based aggregators
    auroc_data = _evaluate_standard_strategies(
        cached_maps, strategies, noise_level, ignore_index
    )
    
    # Step 2: Evaluate the GMM spatial aggregator
    gmm_result = _evaluate_gmm_strategy(
        cached_maps, noise_level, dataset_name, task, variation, decomp
    )
    if gmm_result:
        auroc_data.append(gmm_result)
    
    # Step 3: Combine and finalize the results
    if not auroc_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(auroc_data)
    df = df.sort_values('AUROC', ascending=False).reset_index(drop=True)
    return df