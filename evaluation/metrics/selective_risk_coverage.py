import numpy as np

from tqdm import tqdm
from pathlib import Path 
from typing import List, Any, Tuple, Callable, Dict
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
    
    # Get shared data
    uq_path = shared['paths'].uq_maps
    gt = shared['gt']
    task = shared['task']
    model_noise = shared['model_noise']
    uq_method = shared['uq_method']
    decomp = shared['decomp']
    variation = shared['variation']
    data_noise = shared['data_noise']
    dataset_name = shared['dataset_name']
    ind_to_rem = shared['ind_to_rem']
        
    # Process the strategy
    print(f"Processing aggregator function {strategy_idx}")
    aggr_unc, _ = process_aggr_unc(
        uq_path, gt, task, model_noise, uq_method, decomp, variation, data_noise, 
        method, param, category, ind_to_rem, dataset_name, shared['paths'].metadata
    )
    return strategy_idx, aggr_unc

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

def compute_selective_risks_coverage(gt_list: List[np.ndarray], 
        pred_list: List[np.ndarray],
        paths: Path,  
        task: str, 
        model_noise: int, 
        uq_method: str, 
        decomp: str, 
        variation: str, 
        data_noise: str, 
        strategies: Dict[str, Dict[str, Tuple[callable, Any]]],
        num_workers: int = 4,
        dataset_name: str = 'arctique'
    ) -> None:
    """
    Calculate selective risk-coverage curves for different aggregation strategies.
    
    Args:
        gt_list: list of gt masks
        pred_list: instance and semantic predictions list
        uq_path: path to uncertainty maps
        task: Task type (e.g., "semantic" or "instance")
        model_noise: Image noise level (OOD severity)
        uq_method: UQ method
        decomp: unc. decomposition absed on information theory (e.g. "pu", "au", "eu")
        variation: variation type ingected in data (for OOD severity, e.g. "blood_cells" or "nuclei_intensity")
        data_noise: Mask noise level (if any, seen during training)
        strategies: aggregation strategies dictionary
        num_workers: no. of workers for parallel processing 
        dataset_name: selected dataset
    """
    
    idx_task = 1 if task == 'semantic' else 2
    class_names = CLASS_NAMES_ARCTIQUE if dataset_name.startswith("arctique") else CLASS_NAMES_LIZARD
    total_subkeys = sum(len(subdict) for subdict in strategies.values()) # Count total number of strategies
    
    # Exclude images containing only background (class 0) and preprocess gt masks 
    ind_to_rem, gt_list, pred_list = remove_background_only_images(gt_list, pred_list, idx_task, task, dataset_name)
    gt_list_shared = _process_gt_masks(gt_list, idx_task, dataset_name)

    # Initialize arrays for storing results
    aggr_unc_val = np.zeros((len(pred_list), total_subkeys))
    aggr_acc = np.zeros((len(pred_list), total_subkeys))
    
    # Create list of strategies to process
    strategy_list = []
    idx = 0
    shared_data = {
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
            strategy_list.append((idx, method, param, shared_data, category, method_name))
            idx += 1
    
    # Process strategies in parallel
    aurc_res = {
        'aurc': np.zeros((len(strategy_list))),
        'coverages': np.zeros((len(pred_list) + 1)),
        'selective_risks': np.zeros((len(pred_list) + 1, len(strategy_list)))
        }
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_strategy, data) for data in strategy_list]
        
        for future in tqdm(futures, desc="Processing aggregation strategies"):
            idx, aggr_unc = future.result()
            aggr_acc_val = acc_score(
                gt_list, 
                np.stack(pred_list, axis=0), 
                list(class_names.keys()), 
                len(class_names), 
                shared_data
            )
            
            valid_mask = np.isnan(aggr_acc_val)
            aggr_acc[:, idx] = np.where(valid_mask, 0, aggr_acc_val)
            aggr_unc_val[:, idx]  = np.where(valid_mask, 0, aggr_unc)
            
            evaluator = StatsCache(-aggr_unc_val[:, idx], aggr_acc[:, idx], 10)
            aurc_res['aurc'][idx] = evaluator.aurc/AURC_DISPLAY_SCALE
            selective_risks = _pad_selective_risks(evaluator.selective_risks, pred_list) #TODO - check why for threshold aggregations for softmax we get less selective risks values 
            aurc_res['selective_risks'][:, idx] = selective_risks
    aurc_res['coverages'] = evaluator.coverages
    
    return aurc_res