import sys
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any, Tuple, Dict, NamedTuple, Callable

from evaluation.constants import (CLASS_NAMES_ARCTIQUE, 
                       CLASS_NAMES_LIZARD, 
                       AUROC_STRATEGIES, 
                       BACKGROUND_FREE_STRATEGIES, 
                       AURC_DISPLAY_SCALE, 
                       COLORS)
from evaluation.metrics.accuracy_metrics import per_tile_metrics
from evaluation.data_utils import (DataPaths,
                                   AnalysisResults,
                                   setup_paths, 
                                   load_predictions, 
                                   load_dataset, 
                                   load_unc_maps, 
                                   inst_to_3c)
from evaluation.visualization.plot_functions import create_selective_risks_coverage_plot
from fd_shifts.analysis.metrics import StatsCache
from aggrigator.uncertainty_maps import UncertaintyMap
    
# ---- Configuration Functions ----

def clear_csv_file(output_path: Path, args: argparse.Namespace) -> None:
    """Clears the content of the CSV file if it exists."""
    csv_file = output_path.joinpath(
        f'tables/aurc_data_{args.aggregator_type}_aggr_multi_uq_methods_{args.task}_{args.variation}_{args.data_mod}.csv'
    )
    # Ensure directory exists
    csv_file.parent.mkdir(exist_ok=True, parents=True)
    
    if csv_file.exists():
        csv_file.open('w').close()  # Open in write mode to clear contents
        print(f"Cleared content of {csv_file}")
    else:
        print(f"{csv_file} does not exist yet.")
        
def setup_plot_style() -> None:
    """Sets up the plot style using tueplots and custom configurations."""
    plt.rcParams["text.latex.preamble"] += (
        r"\usepackage{amsmath} \usepackage{amsfonts} \usepackage{bm}"
    )
    # plt.style.use('seaborn-v0_8-whitegrid')
    # plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

def parse_args():
    parser = argparse.ArgumentParser(description='Create accuracy-rejection curves for aggregators')
    parser.add_argument('--task', type=str, default='semantic', help='Task type (e.g., instance or semantic)')
    parser.add_argument('--variation', type=str, help='Variation type (e.g., nuclei_intensity or blood_cells)')
    parser.add_argument('--uq_path', type=str, default='/home/vanessa/Documents/data/uncertainty_arctique_v1-0-corrected_14/', help='Path to unc. evaluation results')
    #arctique path: .../uncertainty_arctique_v1-0-corrected_14/; lizard path:  ../uncertainty_lizard_convnextv2_tiny_3
    parser.add_argument('--label_path', type=str, default='/home/vanessa/Desktop/synth_unc_models/data/v1-0-variations/variations/', help='Path to labels')
    #arctique path: .../Desktop/synth_unc_models/data/v1-0-variations/variations/; lizard path:  ../Documents/synthetic_uncertainty/data/LizardData/
    parser.add_argument('--model_noise', type=int, default=0, help='Mask noise level with which the model was trained')
    parser.add_argument('--image_noise', type=str, default='0_00', help='Image noise level on which the model is evaluated')
    parser.add_argument('--uq_methods', type=str, default='tta,softmax,ensemble,dropout', help='Comma-separated list of UQ methods to evaluate')
    parser.add_argument('--decomp', type=str, default='pu', help='Decomposition component (e.g. pu, au, eu)')
    parser.add_argument('--data_mod', type=str, default='ood', help='Data Modality (e.g. ood or id)')
    parser.add_argument('--aggregator_type', type=str, default='non-pi', help='Aggregator Property (e.g. proportion-invariant or non-pi)' )
    parser.add_argument('--num_workers', type=int, default=4, help='No. of workers for parallel processing' )
    parser.add_argument('--dataset_name', type=str, default='arctique', help='Selected dataset (e.g. arctique or lizard)' )
    
    return parser.parse_args()

# ---- Analysis Functions ----

def process_aggr_unc(uq_path: Path, 
                     gt_sem: np.ndarray, 
                     task: str, 
                     model_noise: int, 
                     uq_method: str, 
                     decomp: str, 
                     variation: str, 
                     data_noise: str, 
                     method: callable, 
                     param: Any,
                     category: str, 
                     ind_to_rem: List,
                     rescale_fact: float):
    """Aggregate uncertainty values with aggrigators' methods"""      
    # Load uncertainty maps
    uq_maps = load_unc_maps(uq_path, task, model_noise, variation, data_noise, uq_method, decomp, True)
    if uq_method != 'softmax' :
        uq_maps = [uqmap/rescale_fact for uqmap in uq_maps]
    uq_maps = [uqmap for i, uqmap in enumerate(uq_maps) if i not in ind_to_rem]
    uq_maps = [UncertaintyMap(array=array, mask=gt, name=None) for array, gt in zip(uq_maps, gt_sem)]
    
    # Apply aggregation method to each ma
    if category == 'Class-based': 
        res = [method(map, param, True) for map in uq_maps]
        return zip(*res)
    res = [method(map, param) for map in uq_maps]
    if category == 'Threshold':
        res = [np.nan_to_num(np.array(res), nan=0)]
    return res, None 

def acc_score(acc_y: np.ndarray, 
        acc_preds: np.ndarray, 
        aggr_w: List[Dict[str, float]], 
        classes_names: List[str], 
        num_classes: int,
        task: str,
    ) -> Tuple[float, Dict[str, float]]:
    """Calculate weighted accuracy scores for panoptic models
    
    Args:
        acc_y: gt masks
        acc_preds: non-rejected predictions 
        aggr_w: aggregtion weights for each class
        classes_names: classes names
        num_classes: no. of classes
        
    Returns:
        Tuple containing weighted-mean F1 score and class-specific average F1 scores
    """
    # Determine which accuracy metric to use 
    acc_metric = "F1" #if classes_names[4].startswith("connective") else "acc"
    
    # Calculation of metrics for each individual tile - images can also be cropped 
    metrics = per_tile_metrics(acc_y, acc_preds, classes_names, num_classes)
    
    if task == 'semantic':
        # Extract F1 score per class for each image
        class_f1_img = {}
        for entry in metrics:
            if entry["class"] != "all":  
                img_id = entry["id"]
                if img_id not in class_f1_img:
                    class_f1_img[img_id] = {}
                class_f1_img[img_id][entry["class"]] = entry[acc_metric] 
    
        f1_cl_avg_per_img = np.array([np.nanmean(list(d.values())) for d in class_f1_img.values()])
        f1_cl_per_img =  np.array([list(d.values()) for d in class_f1_img.values()])
        # weights = np.array([list(d.values()) for d in aggr_w])
        # print(class_f1_img, f1_cl_per_img, f1_cl_per_img.mean(-1), f1_cl_avg_per_img)
        # f1_cl_per_img = np.nan_to_num(f1_cl_per_img, nan=0)
        # f1_cl_avg_per_img = (f1_cl_per_img * weights).sum(-1)
        # print(f1_cl_avg_per_img)
            
        # f1_cls_per_img = np.array([list(d.values()) for d in class_f1_img.values()])
        # weights = [list(d.values()) for d in aggr_w]
        # for i, w in enumerate(weights):
        #     print(f"Length of sublist {i}: {len(w)}")
        return f1_cl_avg_per_img
    else: 
        f1_scores = [entry[acc_metric] for entry in metrics if entry["class"] == "all"]
        return f1_scores

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
    uq_path = shared['uq_path']
    gt_sem = shared['gt_sem']
    task = shared['task']
    model_noise = shared['model_noise']
    uq_method = shared['uq_method']
    decomp = shared['decomp']
    variation = shared['variation']
    data_noise = shared['data_noise']
    dataset_name = shared['dataset_name']
    ind_to_rem = shared['ind_to_rem']
    
    if task == 'instance':
        rescale_fact = np.log(3)
    elif task == 'semantic' and dataset_name.startswith('arctique'):
        rescale_fact = np.log(6)
    elif task == 'semantic' and dataset_name.startswith('lizard'):
        rescale_fact = np.log(7)
    
    class_names = CLASS_NAMES_ARCTIQUE if dataset_name.startswith("arctique") else CLASS_NAMES_LIZARD
    
    # Process the strategy
    print(f"Processing aggregator function {strategy_idx}")
    aggr_unc, weights = process_aggr_unc(
        uq_path, gt_sem, task, model_noise, 
        uq_method, decomp, variation, data_noise, 
        method, param, category, ind_to_rem, 
        rescale_fact
    )
    #We consider avg. sem. F1 for each aggr., but it is possible through the function to compute an imbalance-weighted F1
    # if weights is not None and method_name != 'imbalance-weighted average': 
    #     # Convert weights to use class names instead of indices
    #     INDEX_TO_CLASS = {v: k for k, v in class_names.items()}
    #     transformed_weights = [{INDEX_TO_CLASS[key]: value for key, value in entry.items()} 
    #                         for entry in weights]
    #     return strategy_idx, aggr_unc, transformed_weights
    transformed_weights = [{k: (1 /(len(class_names))) #if k != ('plasma-cell') else 0) 
                            for k, _ in class_names.items()}
                           for _ in range(len(aggr_unc))]
    return strategy_idx, aggr_unc, transformed_weights

def acc_rej(gt_list: List[np.ndarray], 
        pred_list: List[np.ndarray],
        uq_path: Path,  
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
    Calculate accuracy-rejection curves for different aggregation strategies.
    
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
    
    # gt_list = [label.numpy().squeeze() for _ , label in uq_dataset] # Extract gt labels
    portion_vals = np.linspace(0, 1, 50, endpoint=True) # Create evenly spaced rejection portions

    total_subkeys = sum(len(subdict) for subdict in strategies.values()) # Count total number of strategies
    
    # Exclude images containing only bg
    ind_to_rem = [i for i, gt in enumerate(gt_list) if all(class_id == 0 for class_id in np.unique(gt[..., idx_task]))] 
    gt_list = np.array([gt for i, gt in enumerate(gt_list) if i not in ind_to_rem])
    pred_list = [pred for i, pred in enumerate(pred_list) if i not in ind_to_rem]
    if task == 'instance':
        pred_list = [np.stack((inst_to_3c(pred[...,0], False), pred[...,1]), axis = -1) for pred in pred_list]
    print(f"⚠ Removed from evaluation image no. {ind_to_rem} containing only background")
    
    class_names = CLASS_NAMES_ARCTIQUE if dataset_name.startswith("arctique") else CLASS_NAMES_LIZARD

    # Initialize arrays for storing results
    aggr_unc_val = np.zeros((len(pred_list), total_subkeys))
    aggr_acc = np.zeros((len(pred_list), total_subkeys))
    
    # Create list of strategies to process
    strategy_list = []
    idx = 0
    shared_data = {
        'uq_path': uq_path,
        'gt_sem': gt_list[..., idx_task],
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
    aggr_results = {}
    augrc_res = {'augrc_val': np.zeros((len(strategy_list))),  
                 'coverages': np.zeros((len(pred_list) + 1)),
                 'generalized_risks': np.zeros((len(pred_list) + 1, len(strategy_list)))
                }
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_strategy, data) for data in strategy_list]
        
        for future in tqdm(futures, desc="Processing aggregation strategies"):
            idx, aggr_unc, weights = future.result()
            aggr_results[idx] = weights
            aggr_acc_val = acc_score(
                gt_list, np.stack(pred_list, axis=0), weights, 
                list(class_names.keys()), len(class_names), shared_data['task']
            )
            valid_mask = np.isnan(aggr_acc_val)
            aggr_acc[:, idx] = np.where(valid_mask, 0, aggr_acc_val)
            aggr_unc_val[:, idx]  = np.where(valid_mask, 0, aggr_unc)
                        
            evaluator = StatsCache(- aggr_unc_val[:, idx], aggr_acc[:, idx], 10)
            augrc_res['augrc_val'][idx] = evaluator.aurc/AURC_DISPLAY_SCALE
            coverage = evaluator.coverages
            if coverage.shape[0] < len(pred_list) + 1:
                coverage = np.append(coverage, 0)
            augrc_res['coverages'][:] = coverage
            risks = evaluator.selective_risks
            if risks.shape[0] < len(pred_list) + 1:
                risks = np.append(risks, risks[-1])
            augrc_res['generalized_risks'][:, idx] = risks
            print(evaluator.aurc/AURC_DISPLAY_SCALE)
            
    return augrc_res
    
def validate_indices(metadata_file_path, dataset, dataset_name):
    if metadata_file_path.exists():
        indices = np.load(metadata_file_path)
        
    if dataset_name.startswith("arctique"):
        dataset_loader = dataset.dataset # Get the names of the samples, if the dataloader was used in the previous evaluation
        if hasattr(dataset_loader, 'sample_names') and (dataset_loader.sample_names == indices).any():
            print('✓ Uncertainty values, predictions and masks indices match')
        else:
            print('⚠ WARNING: Uncertainty values, predictions and masks indices DO NOT match')
    else:
        print('✓ Uncertainty values, predictions and masks indices match')

def run_aurc_evaluation(args: argparse.Namespace, paths: DataPaths) -> None:
    """
    Run the AURC evaluation pipeline.
    
    Args:
        args: Command line arguments
        output_path: Path to save output
    """
    
    # Extract parameters from arguments
    task = args.task
    model_noise = args.model_noise
    image_noise = args.image_noise
    decomp = args.decomp
    aggregator_type = args.aggregator_type
    num_workers = args.num_workers
    dataset_name = args.dataset_name
    variation = args.variation if args.variation else 'LizardData'
    uq_methods = args.uq_methods.split(',')
    
    # Select appropriate strategies based on aggregator type
    strategies = BACKGROUND_FREE_STRATEGIES if aggregator_type == 'proportion-invariant' else AUROC_STRATEGIES
    
    # Load dataset and ground truth
    dataset, gt_list = load_dataset(
        paths.data, 
        image_noise,
        is_ood=(args.data_mod == 'ood'),
        num_workers=num_workers,
        dataset_name=dataset_name
    )
    
    # Extract method names for plotting
    method_names = [method for category in strategies.values() for method in category.keys()]
    print(method_names)
    
    # Store results for all methods
    all_results = {
        "augrc_val": [],
        "coverages": None,
        "generalized_risks": []
    }
     # Process each UQ method
    for uq_method in uq_methods:
        print(f"\n=== Processing UQ method: {uq_method} ===")
        
        # Load prediction data for current UQ method
        pred_list = load_predictions(
            paths,
            model_noise,
            variation,
            image_noise,
            uq_method,
            True
        )
        
        # Load and check metadata indices for consistency
        metadata_type = f"{task}_noise_{model_noise}_{variation}_{image_noise}_{uq_method}_{decomp}_sample_idx.npy"
        metadata_file_path = paths.metadata.joinpath(metadata_type)
        
        validate_indices(metadata_file_path, dataset, dataset_name)
        
        # Analyze uncertainty and generate results
        print(f"Analyzing uncertainty using {aggregator_type} aggregation strategies with {uq_method}")
        results = acc_rej(
            gt_list,
            pred_list,
            paths.uq_maps,
            task,
            model_noise,
            uq_method,
            decomp,
            variation,
            image_noise,
            strategies,
            num_workers,
            dataset_name
        )
        
        # Store results
        all_results["augrc_val"].append(results["augrc_val"])
        if all_results["coverages"] is None:
            all_results["coverages"] = results["coverages"]
        all_results["generalized_risks"].append(results["generalized_risks"])
    
    # Convert lists to numpy arrays for easier computation
    all_results["augrc_val"] = np.array(all_results["augrc_val"])
    all_results["generalized_risks"] = np.array(all_results["generalized_risks"])
    
    # Calculate mean across all UQ methods
    mean_aurc_val = np.mean(all_results["augrc_val"], axis=0)
    mean_selective_risks = np.mean(all_results["generalized_risks"], axis=0)
    std_selective_risks = np.std(all_results["generalized_risks"], axis=0)
    
    # Create final results structure for plotting
    final_results = AnalysisResults(
        mean_aurc_val=mean_aurc_val,
        coverages=all_results["coverages"],
        mean_selective_risks=mean_selective_risks,
        std_selective_risks=std_selective_risks
    )
    
    # Create plot
    create_selective_risks_coverage_plot(method_names, final_results, paths.output, args)

def main():
    # Set up plot style
    setup_plot_style()
    
    # Parse arguments 
    args = parse_args()
    
    #Set paths and make sure output directory exists
    paths = setup_paths(args)
    
    #Clean Excel file for plot
    clear_csv_file(paths.output, args)
    
    # Run evaluation
    run_aurc_evaluation(args, paths)

if __name__ == "__main__":
    main()