import sys
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any, Tuple, Dict, NamedTuple, Callable
from metrics import per_tile_metrics

from constants import (CLASS_NAMES_ARCTIQUE, STRATEGIES, CLASS_STRATEGIES)
from data_utils import load_predictions, load_dataset, load_unc_maps
from analysis_helpers import remove_rejected
from plot_functions import plot_acc_rej

current_dir = Path.cwd()
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))
utils_dir = current_dir.parent / 'AggroUQ' / 'src'
sys.path.append(str(utils_dir))

from aggrigator.uncertainty_maps import UncertaintyMap

# ---- Data Structures ----

@dataclass
class DataPaths:
    """Container for all data paths used in the program."""
    uq_maps: Path
    metadata: Path
    predictions: Path
    data: Path
    output: Path
    
class AnalysisResults(NamedTuple):
    """Container for analysis results."""
    acc_portion: np.ndarray
    cl_acc_portion: List[Dict[str, float]]
    aggr_unc_val: np.ndarray
    
# ---- Configuration Functions ----

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
    parser.add_argument('--variation', type=str, default='blood_cells', help='Variation type (e.g., nuclei_intensity or blood_cells)')
    parser.add_argument('--uq_path', type=str, default='/home/vanessa/Documents/data/uncertainty_arctique_v1-0-corrected_14/', help='Path to unc. evaluation results')
    parser.add_argument('--label_path', type=str, default='/home/vanessa/Desktop/synth_unc_models/data/v1-0-variations/variations/', help='Path to labels')
    parser.add_argument('--model_noise', type=int, default=0, help='Mask noise level with which the model was trained')
    parser.add_argument('--image_noise', type=str, default='0_00', help='Image noise level on which the model is evaluated')
    parser.add_argument('--uq_method', type=str, default='tta', help='UQ Model (e.g. softmax, ensemble, tta or mcd)')
    parser.add_argument('--decomp', type=str, default='pu', help='Decomposition component (e.g. pu, au, eu)')
    parser.add_argument('--data_mod', type=str, default='ood', help='Data Modality (e.g. ood or id)')
    parser.add_argument('--aggregator_type', type=str, default='class', help='Aggregator Category (e.g. summary or class)' )
    parser.add_argument('--num_workers', type=int, default=4, help='No. of workers for parallel processing' )
    
    return parser.parse_args()

def setup_paths(args: argparse.Namespace) -> DataPaths:
    """Create and validate all necessary paths."""
    base_path = Path(args.uq_path)
    uq_maps_path = base_path.joinpath("UQ_maps")
    metadata_path = base_path.joinpath("UQ_metadata")
    preds_path = base_path.joinpath("UQ_predictions")
    data_path = Path(args.label_path).joinpath(args.variation)
    output_dir = Path.cwd().joinpath('output')
    output_dir.mkdir(exist_ok=True)
    
    for path in [uq_maps_path, metadata_path, preds_path, data_path]: # Validate paths
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
    
    return DataPaths(
        uq_maps=uq_maps_path,
        metadata=metadata_path,
        predictions=preds_path,
        data=data_path,
        output=output_dir
    )

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
                     param: Any):
    """Aggregate uncertainty values with aggrigators' methods"""      
    # Load uncertainty maps
    uq_maps = load_unc_maps(uq_path, task, model_noise, variation, data_noise, uq_method, decomp)
    uq_maps = [UncertaintyMap(array=array, mask=gt, name=None) for array, gt in zip(uq_maps, gt_sem)]
    
    # Apply aggregation method to each ma
    res = [method(map, param, True) for map in uq_maps]
    aggr_val, weights = zip(*res)
    return aggr_val, weights 

def acc_score(acc_y: np.ndarray, 
        acc_preds: np.ndarray, 
        aggr_w: List[Dict[str, float]], 
        classes_names: List[str], 
        num_classes: int
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
    # Calculation of metrics for each individual tile - images can also be cropped 
    metrics = per_tile_metrics(acc_y, acc_preds, classes_names, num_classes)
    # f1_scores = [entry["F1"] for entry in metrics if entry["class"] == "all"]
    # return sum(f1_scores) / len(f1_scores)

    # Extract F1 score per class for each image
    class_f1_img = {}
    for entry in metrics:
        if entry["class"] != "all":  
            img_id = entry["id"]
            if img_id not in class_f1_img:
                class_f1_img[img_id] = {}
            class_f1_img[img_id][entry["class"]] = entry["F1"]
    
    # Initialize dictionaries for class averages and mean F1
    class_averages = {key: [] for key in classes_names}
    class_mean_f1 = []
    
    # Calculate weighted F1 scores
    for i, dict1 in class_f1_img.items():
        if i < len(aggr_w):  # Ensure we don't go out of bounds
            dict2 = aggr_w[i]
            product_values = [dict1[key] * dict2[key] for key in dict1 if key in dict2]
            weighted_avg = sum(product_values) 
            class_mean_f1.append(weighted_avg)
            
            # Add the values of dict1 to class_averages for each key
            for key in dict1:
                if key in class_averages:
                    class_averages[key].append(dict1[key])
                    
    # Now calculate the average for each key across all dict1
    class_averages = {key: np.nanmean(class_averages[key]) for key in class_averages}
                    
    # class_mean_f1 = [sum(f1) / len(f1) for f1 in class_f1_img.values()]
    return np.nanmean(np.array(class_mean_f1)), class_averages

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
    strategy_idx, method, param, shared = strategy_data
    
    # Get shared data
    uq_path = shared['uq_path']
    gt_sem = shared['gt_sem']
    task = shared['task']
    model_noise = shared['model_noise']
    uq_method = shared['uq_method']
    decomp = shared['decomp']
    variation = shared['variation']
    data_noise = shared['data_noise']
    
    # Process the strategy
    print(f"Processing aggregator function {strategy_idx}")
    aggr_unc, weights = process_aggr_unc(
        uq_path, gt_sem, task, model_noise, 
        uq_method, decomp, variation, data_noise, 
        method, param
    )
    
    # Convert weights to use class names instead of indices
    INDEX_TO_CLASS = {v: k for k, v in CLASS_NAMES_ARCTIQUE.items()}
    transformed_weights = [{INDEX_TO_CLASS[key]: value for key, value in entry.items()} 
                          for entry in weights]
    
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
        num_workers: int = 4
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
    """
    
    # gt_list = [label.numpy().squeeze() for _ , label in uq_dataset] # Extract gt labels
    portion_vals = np.linspace(0, 1, 50, endpoint=False) # Create evenly spaced rejection portions

    total_subkeys = sum(len(subdict) for subdict in strategies.values()) # Count total number of strategies
    
    # Initialize arrays for storing results
    aggr_unc_val = np.zeros((len(pred_list), total_subkeys))
    acc_portion = np.empty((len(portion_vals), total_subkeys))
    cl_acc_portion = []
    
    # Create list of strategies to process
    strategy_list = []
    idx = 0
    shared_data = {
        'uq_path': uq_path,
        'gt_sem': gt_list[..., 1],
        'task': task,
        'model_noise': model_noise,
        'uq_method': uq_method,
        'decomp': decomp,
        'variation': variation,
        'data_noise': data_noise
    }
    
    for category, methods in strategies.items():
        for _, (method, param) in methods.items():
            strategy_list.append((idx, method, param, shared_data))
            idx += 1
            
    # Process strategies in parallel
    aggr_results = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_strategy, data) for data in strategy_list]
        
        for future in tqdm(futures, desc="Processing aggregation strategies"):
            idx, aggr_unc, weights = future.result()
            aggr_unc_val[:, idx] = aggr_unc
            aggr_results[idx] = weights
    
    # Calculate accuracy-rejection curves for each strategy
    for m in range(total_subkeys):
        for i, portion in enumerate(portion_vals):   
            print(f"Evaluating strategy #{m} with rejection portion {portion:.2f}")
            
            # Remove rejected samples based on uncertainty
            acc_preds, acc_y = remove_rejected(
                np.stack(pred_list, axis=0), 
                gt_list, 
                portion, 
                aggr_unc_val[:, m]
            )
            
            # Calculate accuracy metrics
            acc_portion[i, m], cl_acc = acc_score(
                acc_y, acc_preds, aggr_results[m], 
                list(CLASS_NAMES_ARCTIQUE.keys()), 6
            )
            
            print(f"Resulting accuracy: {acc_portion[i, m]:.4f}; class accuracies: {cl_acc}")
            cl_acc_portion.append(cl_acc)
            
    return AnalysisResults(acc_portion, cl_acc_portion, aggr_unc_val)

def main(args):
    # Set up paths and configuration
    paths = setup_paths(args)
    
    # Extract parameters from arguments
    task = args.task
    model_noise = args.model_noise
    variation = args.variation
    image_noise = args.image_noise
    uq_method = args.uq_method
    decomp = args.decomp
    aggregator_type = args.aggregator_type
    num_workers = args.num_workers
    
    # Select appropriate strategies based on aggregator type
    strategies = STRATEGIES if aggregator_type == 'summary' else CLASS_STRATEGIES
    
    # Load dataset and ground truth
    dataset, gt_list = load_dataset(
        paths.data, 
        image_noise,
        is_ood=(args.data_mod == 'ood'),
        num_workers=num_workers
    )
    
    # Load prediction data
    pred_list = load_predictions(
        paths,
        model_noise,
        variation,
        image_noise,
        uq_method
    )
    
    # Load and check metadata indices for consistency
    metadata_type = f"{task}_noise_{model_noise}_{variation}_{image_noise}_{uq_method}_{decomp}_sample_idx.npy"
    metadata_file_path = paths.metadata.joinpath(metadata_type)
    
    if metadata_file_path.exists():
        indices = np.load(metadata_file_path)
        dataset_loader = dataset.dataset  # Get the dataset from the DataLoader
        if hasattr(dataset_loader, 'sample_names') and (dataset_loader.sample_names == indices).any():
            print('✓ Uncertainty values, predictions and masks indices match')
        else:
            print('⚠ WARNING: Uncertainty values, predictions and masks indices DO NOT match')
    else:
        print(f"⚠ WARNING: Metadata file not found: {metadata_file_path}")
    
    # Analyze uncertainty and generate results
    print(f"Analyzing uncertainty using {aggregator_type} aggregation strategies")
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
        num_workers
    )
    
    # Extract method names for plotting
    if aggregator_type == 'class':
        method_names = list(CLASS_STRATEGIES['Class-based'].keys())
    else:
        raise NotImplementedError
        # method_names = []
        # for category in STRATEGIES:
        #     method_names.extend(list(STRATEGIES[category].keys()))
    
    # Generate and save plot
    print("Generating visualization")
    plot_acc_rej(
        np.linspace(0, 1, 50, endpoint=False),
        results.acc_portion,
        results.cl_acc_portion[:50],  # Use only the first 50 results to match portion_vals
        uq_method,
        task,
        variation,
        paths.output,
        aggregator_type,
        method_names=method_names
    )
    

if __name__ == "__main__":
    setup_plot_style()
    args = parse_args()
    paths = setup_paths(args)
    
    main(args)
    
    

