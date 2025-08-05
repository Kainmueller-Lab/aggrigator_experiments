import os
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from pathlib import Path 
from typing import List, Any, Tuple, Callable, Dict, Optional
from itertools import combinations
from scipy.stats import wilcoxon
from evaluation.data_utils import (
    load_unc_maps, 
    rescale_maps, 
    remove_background_only_images,
    _process_gt_masks, 
    process_aggr_unc
)
from evaluation.constants import CLASS_NAMES_ARCTIQUE, CLASS_NAMES_LIZARD
from concurrent.futures import ThreadPoolExecutor
from evaluation.metrics.accuracy_metrics import acc_score
from evaluation.constants import AURC_DISPLAY_SCALE
from aggrigator.uncertainty_maps import UncertaintyMap
from fd_shifts.analysis.metrics import StatsCache

def _load_and_align_gmm_scores(
    sample_names: List[str],
    gt_list_processed: List[np.ndarray], # Using processed GT to ensure length matches
    gt_labels: Optional[np.ndarray],
    dataset_name: str,
    task: str,
    variation: str,
    decomp: str,
    ood: bool,
    return_one_only: bool,
) -> Optional[np.ndarray]:
    """
    Loads, aligns, and returns the pre-computed GMM scores.
    
    Args:
        sample_names: The list of sample names for the current batch.
        gt_list_processed: The list of ground truth masks after filtering, for alignment.
        dataset_name, task, variation, decomp: Parameters to construct the GMM scores filename.
        ood, return_one_only: Boolean to perform evaluation on id or ood or both simultaneously

    Returns:
        A NumPy array of GMM scores perfectly aligned with the input lists, or None if it fails.
    """
    # Construct the GMM scores filename and path
    scores_filename = f"{task}_{dataset_name}_{variation}_{decomp}_scores_standardize.csv"
    scores_filepath = os.path.join(os.getcwd(), "spatial", "results", scores_filename)
    
    if not os.path.exists(scores_filepath):
        print(f"Warning: GMM scores file not found at {scores_filepath}, skipping GMM AURC calculation.")
        return None

    print("----Loading and aligning GMM scores for AURC----")
    
    gmm_scores_df = pd.read_csv(scores_filepath)
    gmm_scores_df.rename(columns={'Unnamed: 0': 'uq_map_name', 
                                  'ood_score_normalized_all': 'gmm_score',
                                  'ood_score_normalized_magnitude': 'gmm_score_pix',
                                  'ood_score_normalized_spatial': 'gmm_score_spat',}, 
                         inplace=True)
    gmm_scores_df['uq_map_name'] = gmm_scores_df['uq_map_name'].astype(str)

    if sample_names and isinstance(sample_names[0], torch.Tensor):
        processed_sample_names = [str(name.item()) for name in sample_names]
    else:
        processed_sample_names = [str(name) for name in sample_names]

    alignment_df = pd.DataFrame({'uq_map_name': processed_sample_names})
    
    score_columns_to_merge = ['uq_map_name', 'gmm_score', 'gmm_score_pix', 'gmm_score_spat']
    
    if return_one_only:
        # --- PATH 1: Load only ID or only OoD scores ---
        target_ood_label = 1 if ood else 0
        print(f"Mode: Single Modality. Filtering GMM scores for is_ood == {target_ood_label}.")
        
        # Filter the source DataFrame before merging
        gmm_scores_to_merge = gmm_scores_df[gmm_scores_df['is_ood'] == target_ood_label].copy()
        
        # Merge only on the sample name
        final_scores = pd.merge(
            alignment_df,
            gmm_scores_to_merge[score_columns_to_merge],
            on='uq_map_name',
            how='left'
        )
    else:
        # --- PATH 2: Load both ID and OoD, requiring a composite key ---
        print("Mode: Mixed Modality. Aligning on name and OoD status.")
        if gt_labels is None:
            raise ValueError("gt_labels must be provided when return_one_only is False.")
        
        # Add the OoD status from the current batch to our alignment frame
        alignment_df['is_ood'] = gt_labels
        
        # Use the FULL GMM dataframe and merge on the composite key
        final_scores = pd.merge(
            alignment_df,
            gmm_scores_df[['is_ood'] + score_columns_to_merge],
            on=['uq_map_name', 'is_ood'], # This is the composite key
            how='left'
        )

    # Now, check the result of the merge
    if len(final_scores) != len(gt_list_processed):
         print(f"CRITICAL WARNING: Length mismatch after merge. Expected {len(gt_list_processed)}, got {len(final_scores)}. GMM scores will be skipped.")
         return None
         
    # Handle any samples that were in the batch but truly not in the GMM file
    # This should be a small number, if any.
    if final_scores[['gmm_score', 'gmm_score_pix', 'gmm_score_spat']].isnull().values.any():
        nan_counts = final_scores[['gmm_score', 'gmm_score_pix', 'gmm_score_spat']].isnull().sum()
        print(f"Warning: {nan_counts.to_dict()} samples could not be matched and will be NaN.")
        # We return the full-length array with NaNs and let the calling function handle it. This preserves the array length.
    
    print(f"Successfully prepared {len(final_scores)} GMM scores for alignment.")
    
    # Return the full-length numpy arrays, which may contain NaNs
    return (
        final_scores['gmm_score'].to_numpy(),
        final_scores['gmm_score_pix'].to_numpy(),
        final_scores['gmm_score_spat'].to_numpy()
    )

def process_strategy(
        strategy_data: Tuple[int, Callable, Any, Dict[str, Any]]
    ) -> Tuple[int, np.ndarray, List[Dict[str, float]]]:
    """
    Process a single aggregation strategy.
    Args:
        strategy_data: Tuple containing strategy index, method, parameters and shared data
    Returns:
        Tuple containing strategy index, aggregated uncertainty values and weights
    """
    strategy_idx, method, param, shared, category, method_name = strategy_data
    
    # --- Special handling for GMM placeholder ---
    if method is None or method_name in ['GMM', 'GMM_pixel', 'GMM_spatial']:
        # Return a dummy array of zeros. This will be replaced later by the main function.
        num_samples = len(shared['uq_maps'])
        return strategy_idx, np.zeros(num_samples)
    
    # Get shared data
    uq_maps = shared['uq_maps']
        
    # Process the strategy
    print(f"Processing aggregator function {strategy_idx}")
    
    # Apply aggregation method to each map
    if category == 'Context-aware':
        res = [method(map, param, True) for map in uq_maps]
        # Convert numpy types to Python types for consistency - in some cases the resulting values have a weird np.float(64) format..
        converted_res = []
        for item in res:
            if hasattr(item, 'tolist'):
                converted_res.append(item.tolist())
            elif isinstance(item, (list, tuple)):
                converted_res.append([x.tolist() if hasattr(x, 'tolist') else x for x in item])
            else:
                converted_res.append(item)
        return strategy_idx, list(converted_res)

    res = [method(map, param) for map in uq_maps]

    # Convert numpy types to regular Python types for all cases
    if hasattr(res[0], 'tolist'):
        # If elements are numpy arrays
        res = [item.tolist() if not isinstance(item, (int, np.integer)) else item for item in res]
    elif any(hasattr(item, 'item') for item in res):
        # If elements are numpy scalars
        res = [item.item() if hasattr(item, 'item') else item for item in res]
    if category == 'Threshold':
        res = np.nan_to_num(np.array(res), nan=0).tolist()
    return strategy_idx, res

# --- Bootstrapping and Statistical Testing ---

def _perform_pairwise_wilcoxon_tests_on_eaurc(
    bootstrap_samples_by_agg: Dict[str, List[float]],
) -> pd.DataFrame:
    """Performs one-sided pairwise Wilcoxon signed-rank tests to compare E-AURC."""
    p_value_results = []
    aggr_names = list(bootstrap_samples_by_agg.keys())

    for agg1_name, agg2_name in combinations(aggr_names, 2):
        samples1 = bootstrap_samples_by_agg[agg1_name]
        samples2 = bootstrap_samples_by_agg[agg2_name]
        min_len = min(len(samples1), len(samples2))
        if min_len == 0: continue

        # Test H1: agg1 < agg2 (lower E-AURC is better)
        _, p_less = wilcoxon(samples1[:min_len], samples2[:min_len], alternative='less', zero_method='zsplit')
        # Test H1: agg2 < agg1
        _, p_greater = wilcoxon(samples1[:min_len], samples2[:min_len], alternative='greater', zero_method='zsplit')

        p_value_results.append({
            'Comparison': f'{agg1_name}_vs_{agg2_name}',
            f'p_value ({agg1_name} < {agg2_name})': p_less,
            f'p_value ({agg2_name} < {agg1_name})': p_greater,
        })
    return pd.DataFrame(p_value_results)

def _compute_bootstrapped_aurc_stats(
    aggr_unc_val: np.ndarray,
    aggr_acc_val: np.ndarray,
    strategy_names: List[str],
    n_bootstraps: int = 500
) -> Tuple[Dict, Dict[str, List[float]]]:
    """
    Computes AURC, E-AURC, and Selective Risks with bootstrapping.
    """
    n_samples, n_strategies = aggr_unc_val.shape
    
    # Initialize storage for bootstrapped metrics
    bootstrapped_aurcs = {name: [] for name in strategy_names}
    bootstrapped_eaurcs = {name: [] for name in strategy_names}
    bootstrapped_risks = {name: [] for name in strategy_names}

    print(f"--- Starting bootstrapping with {n_bootstraps} samples ---")
    for _ in tqdm(range(n_bootstraps), desc="Bootstrapping AURC/E-AURC"):
        indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
        
        boot_unc = aggr_unc_val[indices, :]
        boot_acc = aggr_acc_val[indices, :]

        for i, name in enumerate(strategy_names):
            # Ignore strategies that resulted in all NaNs
            if np.isnan(boot_unc[:, i]).all(): continue

            evaluator = StatsCache(-boot_unc[:, i], boot_acc[:, i], 10)
            
            bootstrapped_aurcs[name].append(evaluator.aurc / AURC_DISPLAY_SCALE)
            bootstrapped_eaurcs[name].append(evaluator.eaurc / AURC_DISPLAY_SCALE)
            
            # Pad risks to a consistent length for aggregation
            risks = evaluator.selective_risks
            target_len = n_samples + 1
            if len(risks) < target_len:
                padding = np.full(target_len - len(risks), risks[-1] if len(risks) > 0 else 0)
                risks = np.concatenate([risks, padding])
            bootstrapped_risks[name].append(risks[:target_len])
    
    # Calculate final stats
    results = {
        "mean_aurc": [], "std_aurc": [],
        "mean_eaurc": [], "std_eaurc": [],
        "mean_selective_risks": [], "std_selective_risks": [],
        "coverages": []
    }
    
    for name in strategy_names:
        results["mean_aurc"].append(np.mean(bootstrapped_aurcs[name]) if bootstrapped_aurcs[name] else np.nan)
        results["std_aurc"].append(np.std(bootstrapped_aurcs[name]) if bootstrapped_aurcs[name] else np.nan)
        results["mean_eaurc"].append(np.mean(bootstrapped_eaurcs[name]) if bootstrapped_eaurcs[name] else np.nan)
        results["std_eaurc"].append(np.std(bootstrapped_eaurcs[name]) if bootstrapped_eaurcs[name] else np.nan)
        
        if bootstrapped_risks[name]:
            risk_array = np.array(bootstrapped_risks[name])
            results["mean_selective_risks"].append(np.mean(risk_array, axis=0))
            results["std_selective_risks"].append(np.std(risk_array, axis=0))
        else: # Handle case where a strategy failed completely
            dummy_risks = np.full(n_samples + 1, np.nan)
            results["mean_selective_risks"].append(dummy_risks)
            results["std_selective_risks"].append(dummy_risks)

    # Use the coverages from the last run on the full dataset as representative
    evaluator = StatsCache(-aggr_unc_val[:, 0], aggr_acc_val[:, 0], 10)
    for _ in range(len(results["mean_selective_risks"])):
        results["coverages"].append(evaluator.coverages)
        
    return results, bootstrapped_eaurcs

def _pad_selective_risks(selective_risks, pred_list):
    selective_risks = selective_risks
    target_length = len(pred_list) + 1
    if len(selective_risks) < target_length:
        # Pad with the last value; TODO: check with Carsten why this happens 
        last_value = selective_risks[-1] if len(selective_risks) > 0 else 0
        padding_needed = target_length - len(selective_risks)
        selective_risks = np.concatenate([
            selective_risks, 
            np.full(padding_needed, last_value)
        ])       
    return selective_risks

def compute_uncertainty_and_accuracy_scores(
    uq_maps: List[np.ndarray],
    gt_list: List[np.ndarray],
    pred_list: List[np.ndarray],
    sample_names: List[str],
    gt_labels: np.ndarray,
    paths: Path,
    task: str,
    model_noise: int,
    uq_method: str,
    decomp: str,
    variation: str,
    data_noise: str,
    strategies: Dict[str, Dict[str, Tuple[callable, Any]]],
    num_workers: int = 4,
    dataset_name: str = 'arctique',
    ood: bool = False,
    return_one_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
    """
    Calculates raw, per-image uncertainty and accuracy scores for all strategies.
    This is the core data needed for subsequent bootstrapping.
    """
    idx_task = 1 if task == 'semantic' else 2
    class_names = CLASS_NAMES_ARCTIQUE if dataset_name.startswith("arctique") else CLASS_NAMES_LIZARD
    
    total_subkeys = sum(len(subdict) for subdict in strategies.values())
    
    ind_to_rem, gt_list, pred_list = remove_background_only_images(gt_list, pred_list, idx_task, task, dataset_name)
    
    sample_names = [name for i, name in enumerate(sample_names) if i not in ind_to_rem]
    uq_maps = [map for i, map in enumerate(uq_maps) if i not in ind_to_rem]
    gt_labels = gt_labels if ind_to_rem is None else np.delete(gt_labels, ind_to_rem)
    gt_list_shared = _process_gt_masks(gt_list, idx_task, dataset_name)

    aligned_gmm_scores, aligned_gmm_pixel_scores, aligned_gmm_spatial_scores = _load_and_align_gmm_scores(
        sample_names, gt_list_shared, gt_labels, dataset_name, task, variation, decomp, ood, return_one_only
    )

    aggr_unc_val = np.zeros((len(pred_list), total_subkeys))
    aggr_acc_val = np.zeros((len(pred_list), total_subkeys))
    
    shared_data = {'uq_maps': uq_maps, 'task': task, 'dataset_name': dataset_name}
    
    strategy_list, gmm_strategy_indices, strategy_names_ordered = [], {}, []
    idx = 0
    for category, methods in strategies.items():
        for method_name, (method, param) in methods.items():
            strategy_names_ordered.append(method_name)
            if method_name in ['GMM', 'GMM_pixel', 'GMM_spatial']:
                gmm_strategy_indices[method_name] = idx
            strategy_list.append((idx, method, param, shared_data, category, method_name))
            idx += 1

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_strategy, data) for data in strategy_list]
        for future in tqdm(futures, desc=f"Processing strategies for {uq_method}"):
            idx, aggr_unc = future.result()
            method_name = strategy_list[idx][-1]
            
            gmm_mapping = {'GMM': aligned_gmm_scores, 'GMM_pixel': aligned_gmm_pixel_scores, 'GMM_spatial': aligned_gmm_spatial_scores}
            if method_name in gmm_mapping:
                scores_array = gmm_mapping[method_name]
                if scores_array is not None:
                    aggr_unc = scores_array if len(aggr_unc) == len(scores_array) else np.full(len(aggr_unc), np.nan)
                else:
                    aggr_unc = np.full(len(aggr_unc), np.nan)
            
            aggr_unc_val[:, idx] = aggr_unc
            aggr_acc_val[:, idx] = acc_score(
                gt_list, [pred_list[i] for i in range(len(gt_list))], 
                list(class_names.keys()), len(class_names), shared_data
                )
    
            valid_mask = np.isnan(aggr_acc_val)
            aggr_acc_val = np.where(valid_mask, 0, aggr_acc_val)
            aggr_unc_val = np.where(valid_mask, 0, aggr_unc_val) 

    repro_df = None
    if sample_names:
        repro_df = pd.DataFrame(aggr_unc_val, columns=strategy_names_ordered)
        repro_df['uq_map_name'] = [name.item() if hasattr(name, 'item') else name for name in sample_names]
        repro_df['is_ood'] = gt_labels

    return aggr_unc_val, aggr_acc_val, repro_df