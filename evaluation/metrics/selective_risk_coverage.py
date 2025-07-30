import os
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from pathlib import Path 
from typing import List, Any, Tuple, Callable, Dict, Optional
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
    gmm_scores_df.rename(columns={'Unnamed: 0': 'uq_map_name', 'ood_score_normalized_all': 'gmm_score'}, inplace=True)
    gmm_scores_df['uq_map_name'] = gmm_scores_df['uq_map_name'].astype(str)

    if sample_names and isinstance(sample_names[0], torch.Tensor):
        processed_sample_names = [str(name.item()) for name in sample_names]
    else:
        processed_sample_names = [str(name) for name in sample_names]

    alignment_df = pd.DataFrame({'uq_map_name': processed_sample_names})
    
    if return_one_only:
        # --- PATH 1: Load only ID or only OoD scores ---
        target_ood_label = 1 if ood else 0
        print(f"Mode: Single Modality. Filtering GMM scores for is_ood == {target_ood_label}.")
        
        # Filter the source DataFrame before merging
        gmm_scores_to_merge = gmm_scores_df[gmm_scores_df['is_ood'] == target_ood_label].copy()
        
        # Merge only on the sample name
        final_scores = pd.merge(
            alignment_df,
            gmm_scores_to_merge[['uq_map_name', 'gmm_score']],
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
            gmm_scores_df[['uq_map_name', 'is_ood', 'gmm_score']],
            on=['uq_map_name', 'is_ood'], # This is the composite key
            how='left'
        )

    # Now, check the result of the merge
    if len(final_scores) != len(gt_list_processed):
         print(f"CRITICAL WARNING: Length mismatch after merge. Expected {len(gt_list_processed)}, got {len(final_scores)}. GMM scores will be skipped.")
         return None
         
    # Handle any samples that were in the batch but truly not in the GMM file
    # This should be a small number, if any.
    if final_scores['gmm_score'].isnull().any():
        nan_count = final_scores['gmm_score'].isnull().sum()
        print(f"Warning: {nan_count} samples could not be matched to a GMM score and will be treated as NaN.")
        # We return the full-length array with NaNs and let the calling function handle it. This preserves the array length.
    
    print(f"Successfully prepared {len(final_scores)} GMM scores for alignment.")
    
    # Return the full-length numpy array, which may contain NaNs
    return final_scores['gmm_score'].to_numpy()

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
    if method_name == 'GMM':
        # Return a dummy array of zeros. This will be replaced later.
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

def compute_selective_risks_coverage(uq_maps: List[np.ndarray],
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
    ) -> None:
    """
    Calculate selective risk-coverage curves for different aggregation strategies.
    
    Args:
        cached_maps : cached UncertaintyMap, masks and predictions objects
        gt_list: list of gt masks
        pred_list: instance and semantic predictions list
        paths: preprocess paths
        task: Task type (e.g., "semantic" or "instance")
        model_noise: Image noise level (OOD severity)
        uq_method: UQ method
        decomp: unc. decomposition absed on information theory (e.g. "pu", "au", "eu")
        variation: variation type ingected in data (for OOD severity, e.g. "blood_cells" or "nuclei_intensity")
        data_noise: Mask noise level (if any, seen during training)
        strategies: aggregation strategies dictionary
        num_workers: no. of workers for parallel processing 
        dataset_name: selected dataset
        ood: boolean for data_mod to perform evaluaiton on id or ood
        return_one_only: boolean to perform evaluation on id or ood only
    """
    
    idx_task = 1 if task == 'semantic' else 2 
    # idx_task = 0 if task == 'semantic' else 1 
    class_names = CLASS_NAMES_ARCTIQUE if dataset_name.startswith("arctique") else CLASS_NAMES_LIZARD
    
    # strategies.setdefault('Spatial', {})['GMM'] = (None, None)
    
    total_subkeys = sum(len(subdict) for subdict in strategies.values()) # Count total number of strategies
    
    # Exclude images containing only background (class 0) and preprocess gt masks 
    ind_to_rem, gt_list, pred_list = remove_background_only_images(gt_list, pred_list, idx_task, task, dataset_name)
    
    # --- Filter sample_names and uq_maps just like gt_list and pred_list ---
    # This ensures all lists are aligned after removing background-only images.
    sample_names = [name for i, name in enumerate(sample_names) if i not in ind_to_rem]
    uq_maps = [map for i, map in enumerate(uq_maps) if i not in ind_to_rem]
    
    gt_labels = gt_labels if ind_to_rem is None else np.delete(gt_labels, ind_to_rem) # Filter gt_labels too!
    gt_list_shared = _process_gt_masks(gt_list, idx_task, dataset_name)

    # --- Load and align GMM scores before the main loop ---
    aligned_gmm_scores = _load_and_align_gmm_scores(
        sample_names, 
        gt_list_shared, 
        gt_labels, 
        dataset_name, 
        task, 
        variation,
        decomp, 
        ood, 
        return_one_only
    )

    # Initialize arrays for storing results
    aggr_unc_val = np.zeros((len(pred_list), total_subkeys))
    aggr_acc = np.zeros((len(pred_list), total_subkeys))
    
    # Create list of strategies to process
    strategy_list = []
    idx = 0
    gmm_strategy_idx = -1 # To store the index of the GMM strategy
    
    shared_data = {
        'uq_maps': uq_maps,
        'paths': paths,
        'gt': gt_list_shared,
        'task': task,
        'model_noise': model_noise,
        'uq_method': uq_method,
        'decomp': decomp,
        'variation': variation,
        'data_noise': data_noise,
        'dataset_name': dataset_name,
        'ind_to_rem': ind_to_rem
    }
    
    for category, methods in strategies.items():
        for method_name, (method, param) in methods.items():
            if method_name == 'GMM':
                gmm_strategy_idx = idx # Found it!
            strategy_list.append((idx, method, param, shared_data, category, method_name))
            idx += 1
    
    # Process strategies in parallel
    aurc_res = {
        'aurc': np.zeros((len(strategy_list))),
        'eaurc': np.zeros((len(strategy_list))),
        'coverages': np.zeros((len(pred_list) + 1)),
        'selective_risks': np.zeros((len(pred_list) + 1, len(strategy_list)))
        }
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_strategy, data) for data in strategy_list]
        
        for future in tqdm(futures, desc="Processing aggregation strategies"):
            idx, aggr_unc = future.result()
            
            # --- Overwrite dummy GMM results with real, aligned scores ---
            if idx == gmm_strategy_idx and aligned_gmm_scores is not None:
                if len(aggr_unc) == len(aligned_gmm_scores):
                    print(f"Overwriting dummy values with aligned GMM scores for strategy index {idx}.")
                    aggr_unc = aligned_gmm_scores
                else:
                    print(f"Warning: Length mismatch between GMM scores ({len(aligned_gmm_scores)}) and predictions ({len(aggr_unc)}). Skipping GMM.")
                    # Setting the GMM score to NaN so it's handled gracefully
                    aggr_unc = np.full(len(aggr_unc), np.nan)
            
            aggr_acc_val = acc_score(
                gt_list, 
                [pred_list[i] for i in range(len(gt_list))], #np.stack(pred_list, axis=0), 
                list(class_names.keys()), 
                len(class_names), 
                shared_data
            )
            
            valid_mask = np.isnan(aggr_acc_val)
            aggr_acc[:, idx] = np.where(valid_mask, 0, aggr_acc_val)
            aggr_unc_val[:, idx]  = np.where(valid_mask, 0, aggr_unc)
            
            evaluator = StatsCache(-aggr_unc_val[:, idx], aggr_acc[:, idx], 10)
            aurc_res['aurc'][idx] = evaluator.aurc/AURC_DISPLAY_SCALE
            aurc_res['eaurc'][idx] = evaluator.eaurc/AURC_DISPLAY_SCALE
            selective_risks = _pad_selective_risks(evaluator.selective_risks, pred_list) #TODO - check why for threshold aggregations for softmax we get less selective risks values 
            aurc_res['selective_risks'][:, idx] = selective_risks
    aurc_res['coverages'] = evaluator.coverages
    
    return aurc_res