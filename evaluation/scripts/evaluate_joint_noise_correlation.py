import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import multiprocessing
import time
import yaml
import argparse
import json

from pathlib import Path
from joblib import Parallel, delayed

from aggrigator.uncertainty_maps import UncertaintyMap
from aggrigator.methods import AggregationMethods as am
from aggrigator.summary import AggregationSummary
from evaluation.constants import AUROC_STRATEGIES

def class_mean(unc_map, param):
    assert unc_map.mask_provided, f"Mask not provided for uncertainty map {unc_map.name}"
    assert param in unc_map.class_indices, f"Invalid class label {param} for uncertainty map {unc_map.name}"
    return np.sum(unc_map.array[unc_map.mask == param], dtype=np.float64) / unc_map.class_volumes[param]
    
def get_id_mask(mask, id):
    return np.where(mask==id, 1, 0)

def class_mean_w_custom_weights(unc_map, param): # param = weights: A dict of weights for each class you want to include.
    """
    Compute the weighted average of class means, allowing for custom weights.
    Parameters:
    - unc_map: An object containing class indices and a method to compute class means.
    - param (dict, optional): A dictionary specifying custom weights for each class.
    Returns:
    - Weighted average of class means.
    """
    assert unc_map.mask_provided, f"Mask not provided for uncertainty map {unc_map.name}"
    weights = param
    class_ids = list(weights.keys())
    # Compute class means
    class_means = {class_id: class_mean(unc_map, class_id)
                    for class_id in class_ids}
    # Ensure provided weights sum to 1
    weight_sum = sum(weights.values())
    if not abs(weight_sum - 1.0) < 1e-6:
        print(f"Warning: Weights do not sum to 1. Sum is {weight_sum}. Difference: {abs(weight_sum - 1.0)}. Weights: {weights}")
    return sum(class_means[id] * weights[id] for id in class_ids)

# Modified aggregation functions that accept ignore_index as parameter
def class_mean_w_equal_weights_configurable(unc_map, param):
    """
    param should be a dict with keys: 'include_background' and 'ignore_index'
    e.g., param = {'include_background': False, 'ignore_index': 0}
    """
    include_background = param.get('include_background', False)
    ignore_index = param.get('ignore_index', 0)
    
    classes = [class_id for class_id in unc_map.class_indices if not (class_id == ignore_index and not include_background)]
    weights = {id: 1 / len(classes) for id in classes}
    return class_mean_w_custom_weights(unc_map, weights)

def class_mean_weighted_by_occurrence_configurable(unc_map, param):
    """
    param should be a dict with keys: 'include_background' and 'ignore_index'
    e.g., param = {'include_background': False, 'ignore_index': 0}
    """
    include_background = param.get('include_background', False)
    ignore_index = param.get('ignore_index', 0)
    
    classes = [class_id for class_id in unc_map.class_indices if not (class_id == ignore_index and not include_background)]
    class_pixel_counts = {class_id: get_id_mask(unc_map.mask, class_id).sum() for class_id in classes}
    fg_pixel_count = np.sum(list(class_pixel_counts.values()))
    weights = {id: class_pixel_counts[id] / fg_pixel_count for id in classes}
    return class_mean_w_custom_weights(unc_map, weights)

# Function to create dataset-specific focus strategy list
def create_focus_strategy_list(ignore_index=0, include_background=False):
    """Create a focus strategy list with dataset-specific ignore_index"""
    
    class_mean_params = {
        'include_background': include_background,
        'ignore_index': ignore_index
    }
    
    return [
        (am.mean, None),
        (am.above_threshold_mean, 0.3),
        (am.above_threshold_mean, 0.5),
        (am.above_threshold_mean, 0.7),
        (am.above_threshold_mean, 0.9),
        (am.above_threshold_mean, 0.95),
        (am.above_quantile_mean, 0.3),
        (am.above_quantile_mean, 0.5),
        (am.above_quantile_mean, 0.7),
        (am.above_quantile_mean, 0.9),
        (am.above_quantile_mean_fg_ratio, None),
        (am.patch_aggregation, 10), 
        (am.patch_aggregation, 20),
        (am.patch_aggregation, 40),
        (am.patch_aggregation, 80),
        (am.patch_aggregation, 100),
        (class_mean_w_equal_weights_configurable, class_mean_params),
        (class_mean_weighted_by_occurrence_configurable, class_mean_params),
    ]

def load_dataset_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_correlation_matrix_plot(corr_matrix, filename, correlation_type, save_dir, save_col = False):
    """
    Computes and plots the correlation matrix of methods with clean names.
    """
    # Create a mapping for clean method names
    name_mapping = {
        "class_mean_w_equal_weights_configurable_{'include_background': False, 'ignore_index': 0}": "class_mean_w_equal_weights",
        "class_mean_w_equal_weights_configurable_{'include_background': False, 'ignore_index': 255}": "class_mean_w_equal_weights",
        "class_mean_weighted_by_occurrence_configurable_{'include_background': False, 'ignore_index': 0}": "class_mean_weighted_by_occurrence",
        "class_mean_weighted_by_occurrence_configurable_{'include_background': False, 'ignore_index': 255}": "class_mean_weighted_by_occurrence",
        # Add more mappings as needed for different configurations
    }
    
    # Apply name mapping to correlation matrix
    corr_matrix_clean = corr_matrix.copy()
    
    # Clean column names
    new_columns = []
    for col in corr_matrix_clean.columns:
        clean_name = col
        for old_name, new_name in name_mapping.items():
            if old_name in col:
                clean_name = new_name
                break
        new_columns.append(clean_name)
    
    # Clean index names
    new_index = []
    for idx in corr_matrix_clean.index:
        clean_name = idx
        for old_name, new_name in name_mapping.items():
            if old_name in idx:
                clean_name = new_name
                break
        new_index.append(clean_name)
    
    corr_matrix_clean.columns = new_columns
    corr_matrix_clean.index = new_index

    # --- Saving colors for unique columns of 'Mean'
    if correlation_type == "spearman" and save_col is True:
        mean_candidates = [col for col in corr_matrix_clean.columns if col == "mean"]
        if not mean_candidates:
            print("No 'Mean' column found in correlation matrix.")
            mean_correlations = {}
        else:
            # Use the first occurrence of 'Mean' column
            mean_correlations = corr_matrix_clean["mean"].iloc[:].to_dict()

        def get_correlation_color(correlation_value):
            """Map correlation value to color based on RdBu_r colormap"""
            normalized = (correlation_value + 1) / 2
            cmap = plt.cm.RdBu_r
            rgba = cmap(normalized)
            return '#{:02x}{:02x}{:02x}'.format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

        method_colors = {}
        for method_name in corr_matrix_clean.index:
            correlation_with_mean = mean_correlations.get(method_name, 0.0)
            if isinstance(correlation_with_mean, dict):
                print(f"Warning: correlation for {method_name} is a dict. Skipping.")
                correlation_with_mean = 0.0
            color = get_correlation_color(correlation_with_mean)


            method_colors[method_name] = {
                'correlation_with_mean': correlation_with_mean,
                'color': color,
            }

        # Save color mapping
        colors_dir = os.path.join(save_dir, "colors")
        os.makedirs(colors_dir, exist_ok=True)
        color_mapping_file = os.path.join(colors_dir, f"{filename}_method_colors.json")
        with open(color_mapping_file, 'w') as f:
            json.dump(method_colors, f, indent=2)
    
    # Plot the correlation matrix as a heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    method_names = corr_matrix_clean.index.tolist()
    
    sns.heatmap(corr_matrix_clean, ax=ax, cmap="RdBu_r", annot=False, fmt=".2f", #RdBu_r #inferno
                cbar=True, vmin=-1, vmax=1, 
                xticklabels=method_names, yticklabels=method_names,
                square=True, linewidths=0.5)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Color strategy names by category
    color_code = {
        "threshold": "#D0F7E3",  # Light green
        "quantile": "#634861",   # Purple
        "patch": "#6A92B8",      # Blue
        "class_mean": "#C59FD6", # Light purple
        "mean": "#FFE5B4",       # Light yellow
        "gmm": "#1b8565",        # Emerald
    }
    
    for tick in ax.get_xticklabels():
        strategy_name = tick.get_text()
        color = "white"
        for key in color_code:
            if key in strategy_name.lower():
                color = color_code[key]
                break
        tick.set_bbox(dict(facecolor=color, edgecolor='none', alpha=0.7, boxstyle="round,pad=0.3"))
    
    for tick in ax.get_yticklabels():
        strategy_name = tick.get_text()
        color = "white"
        for key in color_code:
            if key in strategy_name.lower():
                color = color_code[key]
                break
        tick.set_bbox(dict(facecolor=color, edgecolor='none', alpha=0.7, boxstyle="round,pad=0.3"))

    plt.title(f"Method Correlations: {filename}", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def process_single_dataset(dataset, dataset_name, sample_size, num_workers, ignore_index=0, include_background=False):
    """
    Process a single dataset and return the summary dataframe.
    """
    # Create dataset-specific focus strategy list
    dataset_focus_strategy_list = create_focus_strategy_list(ignore_index, include_background)
    
    sample_size = len(dataset) if sample_size == 0 else sample_size

    # Print info
    dataset_info = dataset.get_info()
    dataset_info.pop('semantic_mapping', None)
    print("____________________")
    print(f"Processing dataset: {dataset_name}")
    for key, value in dataset_info.items():
        print(f"{key}: {value}")
    print(f"Number of samples used: {sample_size} of {len(dataset)}")
    
    if dataset.num_classes is None:
        print(f"WARNING: Could not normalize UQ maps because dataset.num_classes is not defined.")
    else:
        print(f"NOTE: Normalizing UQ maps by ln(K) where K={dataset.num_classes} is the number of classes.")
        # if dataset.num_classes != 2 and (am.above_quantile_mean_fg_ratio, None) in focus_strategy_list:
        #     focus_strategy_list.remove((am.above_quantile_mean_fg_ratio, None))
    print("____________________")

    def aggregate(sample):
        # Load uncertainty maps and predictions from dataset
        prediction = sample['prediction']
        uq_array = sample['uq_map']

        # NOTE: Weedsgalore predictions are 3D arrays with a single channel
        if prediction.ndim == 3 and prediction.shape[0] == 1:
            prediction = prediction.squeeze(0)

        # NOTE: Arctique and Lizard predictions are 3D arrays with two channels
        if prediction.ndim == 3 and prediction.shape[2] == 2:
            prediction = prediction[:, :, -1]

        # Slice if 3D
        if uq_array.ndim == 3:
            print(f"Warning: 3D UQ map detected. Only middle 2D slice are used.")
            mid_slice = uq_array.shape[0] // 2
            uq_array = uq_array[mid_slice, :, :]
            prediction = prediction[mid_slice, :, :]
        
        # Replace negative values with zero
        uq_array = np.where(uq_array < 0, 0, uq_array)

        # Ignore too small images for patch aggregation
        h, w = uq_array.shape
        patch_200_in_agg_list = any(strategy[0] == am.patch_aggregation and strategy[1] == 200 
                                   for strategy in dataset_focus_strategy_list)
        if patch_200_in_agg_list and (h < 200 or w < 200):
            print(f"Warning: Ignoring UQ map {sample['sample_name']} because it is too small.")
            return None
        
        # Normalize arrays by ln(K) where K is number of classes
        if dataset_info['num_classes'] is not None:
            uq_array = uq_array / np.log(dataset_info['num_classes'])

        # Apply aggregation strategies
        uq_map = UncertaintyMap(array=uq_array, mask=prediction, name=sample['sample_name'])
        summary = AggregationSummary(dataset_focus_strategy_list, num_cpus=1)
        return summary.apply_methods([uq_map], save_to_excel=False, do_plot=False, max_value=1.0)
    
    # Aggregate all UQ maps
    start = time.time()
    n_jobs = multiprocessing.cpu_count() if num_workers == 0 else num_workers
    summary_dfs = Parallel(n_jobs=n_jobs, verbose=10)(delayed(aggregate)(dataset[idx]) for idx in range(sample_size))
    summary_dfs = [df.set_index("Name") for df in summary_dfs if df is not None]
    summary_df = pd.concat(summary_dfs, axis=1).reset_index()
    print(f"Computed aggregation strategy summary: {time.time() - start} s")
    
    # Transform to have samples as rows and methods as columns
    summary_df = summary_df.T
    summary_df = summary_df.reset_index()
    summary_df.columns = summary_df.iloc[0]
    summary_df = summary_df.drop(index=0).reset_index(drop=True)
    summary_df.rename(columns={'Name': 'uq_map_name'}, inplace=True)
    
    # Add dataset name and noise level columns
    summary_df.insert(loc=1, column="dataset_name", value=dataset_name)
    
    return summary_df

def load_and_merge_spatial_scores(summary_df, base_dataset_name, noise_level):
    """
    Loads spatial GMM scores and merges them into the summary dataframe.
    This function specifically handles the 'gta' dataset and its unique indexing.
    """
    target_datasets = ['gta', 'lidc', 'arctique', 'weedsgalore', 'ade20k', 'lizard', 'wormbodies']
    
    # Check if the function should run for the current dataset
    current_dataset_match = [name for name in target_datasets if name in base_dataset_name]
    if not current_dataset_match:
        return summary_df
    
    # Get the specific dataset name for logging and logic
    current_dataset = current_dataset_match[0]
    print(f"\n--- Running Spatial Score Merge for {current_dataset.upper()} (Noise Level: {noise_level}) ---")

    parts = base_dataset_name.split('_')
    if current_dataset in ('arctique', 'lizard'):
        task, dataset_name, variation, decomp = parts[3], parts[2], f"{parts[4]}_{parts[5]}", parts[7]
    elif current_dataset == 'weedsgalore':
        task, dataset_name, variation, decomp = f"{parts[3]}_{parts[4]}_{parts[5]}", parts[2], parts[6], parts[8] # Assuming 'none' variation
        # scores_filename = f"{task}_{dataset_name}_{decomp}_scores_standardize.csv" #standardize/identity/beta/quantile 
    elif current_dataset == 'ade20k':
        task, dataset_name, variation, decomp = parts[4], parts[2], parts[5], parts[7] 
    else: # Handles GTA, LIDC
        task, dataset_name, variation, decomp = parts[3], parts[2], parts[4], parts[6]

    # if current_dataset != 'weedsgalore':
    scores_filename = f"{task}_{dataset_name}_{variation}_{decomp}_scores_standardize.csv"
    
    print(f"SCORES_FILENAME {scores_filename}")
        
    scores_filepath = os.path.join(os.getcwd(), "spatial", "results", scores_filename)

    if not os.path.exists(scores_filepath):
        print(f"Warning: Spatial scores file not found at {scores_filepath}. Skipping merge.")
        return summary_df

    scores_df = pd.read_csv(scores_filepath)
    scores_df.rename(columns={'Unnamed: 0': 'sample_key'}, inplace=True)

    if noise_level == '0_00': # In-Distribution
        scores_to_process = scores_df[scores_df['is_ood'] == 0].copy()
        if scores_to_process.empty:
            return summary_df
        
        if 'gta' in base_dataset_name:
            # For GTA, apply 5-digit zero-padding
            scores_to_process['uq_map_name'] = scores_to_process['sample_key'].apply(lambda x: f"{int(x):05d}")
        else:
            # All other datasets (LIDC, Arctique, Weedsgalore, ADE20k) have matching string names
            scores_to_process['uq_map_name'] = scores_to_process['sample_key'].astype(str)
       
    elif noise_level in ['0_25', '0_50', '0_75', '1_00']: # Out-of-Distribution
        scores_to_process = scores_df[scores_df['is_ood'] == 1].copy()
        if scores_to_process.empty:
            return summary_df
        scores_to_process.rename(columns={'sample_key': 'uq_map_name'}, inplace=True)
    else:
        return summary_df
    
    scores_to_merge = scores_to_process[['uq_map_name', 'ood_score_normalized_all']] #ood_score_normalized_spatial, ood_score_normalized_magnitude
    
    # Convert both 'on' columns to strings to guarantee they match.
    summary_df['uq_map_name'] = summary_df['uq_map_name'].astype(str)
    scores_to_merge['uq_map_name'] = scores_to_merge['uq_map_name'].astype(str)
    
    merged_df = pd.merge(
        summary_df,
        scores_to_merge,
        on='uq_map_name',
        how='left'
    )
    
    merged_df.rename(columns={'ood_score_normalized_all': 'gmm_normalized_score'}, inplace=True) #ood_score_normalized_spatial, ood_score_normalized_magnitude
    merged_df['gmm_normalized_score'] = pd.to_numeric(merged_df['gmm_normalized_score'], errors='coerce')
    
    num_valid_scores = merged_df['gmm_normalized_score'].notna().sum()
    print(f"Merge complete. Found and merged {num_valid_scores} valid scores.")
    print("--- End Spatial Score Merge ---\n")
    return merged_df

def compute_individual_noise_correlations(dataset, dataset_name, sample_size, num_workers, base_dataset_name, 
                                          noise_level, ignore_index=0, include_background=False):
    """
    Compute correlations for individual noise levels.
    
    Args:
        dataset: Single dataset for specific noise level
        dataset_name: Name including noise level
        sample_size: Number of samples
        num_workers: Number of workers
        base_dataset_name: Base dataset name for file naming
        noise_level: Noise level for file naming 
        ignore_index: Index to ignore
        include_background: Whether to include background
    """
    # Process single dataset
    summary_df = process_single_dataset(dataset, dataset_name, sample_size, num_workers, ignore_index, include_background)
    
    # The summary_df will now contain the 'gmm_normalized_score' if available.
    # summary_df = load_and_merge_spatial_scores(summary_df, base_dataset_name, noise_level)
    
    # Save individual summary
    out_name = f"aggregation_value_summary_{base_dataset_name}_{noise_level}"
    summary_df.to_csv(os.path.join("output", "tables", f"{out_name}.csv"), index=False) #"joint_correlation", f"{out_name}.csv"), index=False)
    print(f"Individual aggregation value summary {out_name}.csv saved to output folder.")
    
    # # Compute correlations between methods (columns)
    # method_columns = [col for col in summary_df.columns if col not in ['uq_map_name', 'dataset_name']]
    # correlation_df = summary_df[method_columns]
    
    # start = time.time()
    # correlations = {}
    # for correlation_type in ["pearson", "spearman", "kendall"]:
    #     # Compute correlation between methods (columns)
    #     corr_matrix = correlation_df.corr(method=correlation_type, min_periods=1)
    #     correlations[correlation_type] = corr_matrix
    # print(f"Computed individual correlation matrices for {noise_level}: {time.time() - start} s")
    
    # # Save correlation matrices and plots for individual noise level
    # for correlation_type, corr_matrix in correlations.items():
    #     out_name = f"correlation_matrix_{correlation_type}_{base_dataset_name}_{noise_level}"
        
    #     # Save to csv
    #     corr_matrix.to_csv(os.path.join("output", "tables", "joint_correlation", f"{out_name}.csv"))
    #     print(f"Individual correlation matrix {out_name}.csv saved to output folder.")
        
    #     # Save heatmap as png
    #     save_correlation_matrix_plot(corr_matrix, out_name, correlation_type, os.path.join("output", "figures", "joint_correlation"), True)
    #     print(f"Individual correlation heatmap {out_name}.png saved to output folder.")
    
    return summary_df

# Dataset configuration dictionary
DATASET_CONFIGS = {
    'ade20k': {'ignore_index': 0, 'include_background': False},
    'arctique': {'ignore_index': 0, 'include_background': False},
    'lidc': {'ignore_index': 0, 'include_background': False},
    'gta': {'ignore_index': 255, 'include_background': False},
    'weedsgalore': {'ignore_index': 0, 'include_background': False},
    # Add more datasets as needed with their specific configurations
}

def evaluate_correlation_across_noise_levels(datasets_dict, sample_size, num_workers, base_dataset_name, compute_individual=False):
    """
    Evaluate correlations across different noise levels by concatenating data.
    
    Args:
        datasets_dict: Dictionary with noise levels as keys and datasets as values
        sample_size: Number of samples per dataset
        num_workers: Number of parallel workers
        base_dataset_name: Base name for the dataset
        compute_individual: Whether to compute individual noise level correlations
    """
    # Get dataset configuration
    dataset_base_name = base_dataset_name.split('_')[2]  # Extract base name (e.g., 'ade20k' from 'joint_noise_ade20k_...')
    config = DATASET_CONFIGS.get(dataset_base_name, {'ignore_index': 0, 'include_background': False})
   
    all_summary_dfs = []
    
    # Process each dataset (noise level)
    for noise_level, dataset in datasets_dict.items():
        dataset_name = f"{base_dataset_name}_{noise_level}"
        
        if compute_individual:
            # Compute individual noise level correlations
            summary_df = compute_individual_noise_correlations(
                dataset, dataset_name, sample_size, num_workers, base_dataset_name,
                noise_level, config['ignore_index'], config['include_background']
            )
        else:
            # Just process the dataset
            summary_df = process_single_dataset(
                dataset, dataset_name, sample_size, num_workers, 
                config['ignore_index'], config['include_background']
            )
        
        # Add noise level column
        summary_df.insert(loc=2, column="noise_level", value=noise_level)
        all_summary_dfs.append(summary_df)
    
    # Check if there's only one noise level - terminate early if so
    if len(datasets_dict) == 1:
        print(f"Only one noise level found ({list(datasets_dict.keys())[0]}). No across-noise correlation to compute.")
        return
    
    # Concatenate all datasets
    combined_df = pd.concat(all_summary_dfs, ignore_index=True)
    
    # Save combined summary
    out_name = f"aggregation_value_summary_{base_dataset_name}_combined"
    combined_df.to_csv(os.path.join("output", "tables", f"{out_name}.csv"), index=False) #"joint_correlation", f"{out_name}.csv"), index=False)
    print(f"Combined aggregation value summary {out_name}.csv saved to output folder.")
    
    # # FIXED: Compute correlations between methods (columns), not samples (rows)
    # method_columns = [col for col in combined_df.columns if col not in ['uq_map_name', 'dataset_name', 'noise_level']]
    
    # # Use only the method columns for correlation computation
    # correlation_df = combined_df[method_columns]
    
    # start = time.time()
    # correlations = {}
    # for correlation_type in ["pearson", "spearman", "kendall"]:
    #     # Compute correlation between methods (columns)
    #     corr_matrix = correlation_df.corr(method=correlation_type, min_periods=1)
    #     correlations[correlation_type] = corr_matrix
    # print(f"Computed correlation matrices: {time.time() - start} s")
    
    # # Save correlation matrices and plots
    # for correlation_type, corr_matrix in correlations.items():
    #     out_name = f"correlation_matrix_{correlation_type}_{base_dataset_name}_combined"
        
    #     # Save to csv
    #     corr_matrix.to_csv(os.path.join("output", "tables", "joint_correlation", f"{out_name}.csv"))
    #     print(f"Correlation matrix {out_name}.csv saved to output folder.")
        
    #     # Create a temporary dataframe for plotting with method names
    #     plot_df = pd.DataFrame(index=method_columns)
    #     plot_df['Name'] = method_columns
    #     plot_df = plot_df.set_index('Name')
        
    #     # Save heatmap as png
    #     save_correlation_matrix_plot(corr_matrix, out_name, correlation_type, os.path.join("output", "figures", "joint_correlation"), True)
    #     print(f"Correlation heatmap {out_name}.png saved to output folder.")


# Modified dataset creation functions
def create_ade20k_datasets(model_id, uq_method):
    """Create ADE20K datasets for both noise levels."""
    datasets = {}
    noise_levels = ['0_00', '1_00']
    folders = ['validation', 'test_cityscapes']
    split_path =  ["/fast/AG_Kainmueller/data/GTA_ValUES_splits/ADE20k_id_test", None]
    
    for nl, fold, split in zip(noise_levels, folders, split_path):
        extra_info = {
            'task': 'semantic',
            'variation': 'cityscapes',
            'model_noise': 0,
            'data_noise': nl,
            'uq_method': uq_method,
            'decomp': 'pu',
            'spatial': None,
            'split_path': None, #split,
            'split': None,
            'metadata': False,
            'model_checkpoint': model_id,
        }
        
        # Construct the path to the potential new split file
        dynamic_split_filename = f"{extra_info['task']}_ade20k_{extra_info['variation']}_{extra_info['decomp']}_test_split.json"
        dynamic_split_path = os.path.join(os.getcwd(), "spatial", "splits", dynamic_split_filename)
        
        # Check if the dynamic split file exists
        use_dynamic_split = os.path.exists(dynamic_split_path)
        if use_dynamic_split:
            print(f"Found spatial split file. Using: {dynamic_split_path}")
        
        # If it's the '0_00' noise level and the dynamic file exists, override the none split path
        if nl == '0_00' and use_dynamic_split:
            extra_info['split_path'] = dynamic_split_path
       
        image_path = f'/fast/AG_Kainmueller/data/ADEChallengeData2016/images/{fold}'
        mask_path = f'/fast/AG_Kainmueller/data/ADEChallengeData2016/annotations/{fold}'
        uq_map_path = f'/fast/AG_Kainmueller/data/UQ_maps/ADE20K/'
        prediction_path = '/fast/AG_Kainmueller/data/ADEChallengeData2016/'
        
        from datasets.ADE20K.ade20k_dataset_creation import OptimizedADE20K_CityscapesDataset
        dataset = OptimizedADE20K_CityscapesDataset(image_path, mask_path, uq_map_path, prediction_path, 
                                          '/fast/AG_Kainmueller/data/ADEChallengeData2016/objectInfo150.json',
                                          **extra_info)
        dataset.num_classes = 150
        datasets[nl] = dataset
    
    return datasets


def create_arctique_datasets(task, uq_method):
    """Create Arctique datasets for both noise levels."""
    datasets = {}
    noise_levels = ['0_00', '0_25', '0_50', '0_75', '1_00'] #if task == 'semantic' else ['0_00', '0_25']
    
    for noise_level in noise_levels:
        variation = 'blood_cells' if task == 'semantic' else 'nuclei_intensity'
        extra_info = {
            'task': task,
            'variation': variation,
            'model_noise': 0,
            'data_noise': noise_level,
            'uq_method': uq_method,
            'decomp': 'pu',
            'spatial': False,
            'metadata': False, #Note: even if gt does not match with preds and uq_maps it doesn't matter in this context since gt is never used in this script !
            'split_path': None, 
            'split': None,
        }
        
        # # Construct the path to the potential new split file
        # dynamic_split_filename = f"{extra_info['task']}_arctique_{extra_info['variation']}_{extra_info['decomp']}_test_split.json"
        # dynamic_split_path = os.path.join(os.getcwd(), "spatial", "splits", dynamic_split_filename)
        
        # # Check if the dynamic split file exists
        # use_dynamic_split = os.path.exists(dynamic_split_path)
        # if use_dynamic_split:
        #     print(f"Found spatial split file. Using: {dynamic_split_path}")
        
        # # If it's the '0_00' noise level and the dynamic file exists, override the none split path
        # if noise_level == '0_00' and use_dynamic_split:
        #     extra_info['split_path'] = dynamic_split_path
        
        map_path = Path('/fast/AG_Kainmueller/vguarin/hovernext_trained_models/trained_on_cluster/uncertainty_arctique_v1-0-corrected_14')
        base_path = Path('/fast/AG_Kainmueller/synth_unc_models/data/v1-0-variations/variations/')
        prediction_path = map_path.joinpath('UQ_predictions')
        uq_map_path = map_path.joinpath('UQ_maps')

        from datasets.Arctique.arctique_dataset_creation import OptimizedArctiqueDataset, SharedMaskCache
        mask_cache = SharedMaskCache()
        ref_mask_path = base_path.joinpath(extra_info['variation'], '0_00', 'masks')
        ref_image_path = base_path.joinpath(extra_info['variation'], '0_00', 'images')

        sample_names = [int(digits) for filename in os.listdir(ref_image_path)
                       if (digits := ''.join(filter(str.isdigit, filename)))]
        
        shared_masks = mask_cache.get_masks(ref_mask_path, sample_names, extra_info['task'])
        
        dataset = OptimizedArctiqueDataset(ref_image_path, ref_mask_path, uq_map_path, prediction_path, 
                                          'abc', shared_masks, **extra_info)
        dataset.num_classes = 6
        datasets[noise_level] = dataset
    
    return datasets


def create_lidc_datasets(variation, uq_method):
    """Create LIDC datasets for both noise levels."""
    datasets = {}
    noise_levels = ['0_00', '1_00']
    
    # Remove patch aggregation 200 for LIDC
    # global focus_strategy_list
    # if (am.patch_aggregation, 200) in focus_strategy_list:
    #     focus_strategy_list.remove((am.patch_aggregation, 200))
    
    for noise_level in noise_levels:
        extra_info = {
            'task': 'fgbg',
            'variation': variation,
            'model_noise': 0,
            'data_noise': noise_level,
            'uq_method': uq_method,
            'decomp': 'pu',
            'spatial': None,
            'cons_thresh': 2,
            'metadata': False,
            'render_2d': True,
            'render_ind_masks': False,
            'split_path': None, 
            'split': None,
        }
        
        # # Construct the path to the potential new split file
        # dynamic_split_filename = f"{extra_info['task']}_lidc_{extra_info['variation']}_{extra_info['decomp']}_test_split.json"
        # dynamic_split_path = os.path.join(os.getcwd(), "spatial", "splits", dynamic_split_filename)
        
        # # Check if the dynamic split file exists
        # use_dynamic_split = os.path.exists(dynamic_split_path)
        # if use_dynamic_split:
        #     print(f"Found spatial split file. Using: {dynamic_split_path}")
        
        # # If it's the '0_00' noise level and the dynamic file exists, override the none split path
        # if noise_level == '0_00' and use_dynamic_split:
        #     extra_info['split_path'] = dynamic_split_path
        
        base_path = Path('/fast/AG_Kainmueller/data/ValUES/')
        cycle = 'FirstCycle'
        folder = f"{extra_info['variation']}_fold0_seed123"
        placeholder = "Softmax"
        data_path = base_path.joinpath(f"{cycle}/{placeholder}/test_results/{folder}/")
        
        if extra_info['data_noise'] == "0_00":
            data_dir = data_path / "id"
        else:
            data_dir = data_path / "ood"
            
        image_path = data_dir / "input"
        mask_path = data_dir / "gt_seg"
        prediction_path = base_path.joinpath('UQ_predictions')
        uq_map_path = base_path.joinpath('UQ_maps')
        
        from datasets.LIDC.lidc_dataset_creation import LIDCDataset
        dataset = LIDCDataset(image_path, mask_path, uq_map_path, prediction_path, 'abc', **extra_info)
        dataset.num_classes = 2
        datasets[noise_level] = dataset
    
    return datasets

def create_gta_datasets(uq_method):
    """Creates GTA datasets, dynamically checking for a spatial split file."""
    datasets = {}
    noise_levels = ['0_00', '1_00']
    # Define original/default paths
    original_split_paths = ["/fast/AG_Kainmueller/data/GTA_ValUES_splits/GTA_id_test", None]
    folders = ['OriginalData', 'CityScapesOriginalData']
    
    for i, noise_level in enumerate(noise_levels):
        current_split_path = original_split_paths[i]
        fold = folders[i]
        
        extra_info = {
            'task': 'semantic',
            'variation': 'cityscapes',
            'model_noise': 0,
            'data_noise': noise_level,
            'uq_method': uq_method,
            'decomp': 'pu',
            'spatial': None,
            'split_path': current_split_path, #current_split_path, None if dynamic_split_path is used
            'split': None
        }
        
        # # Construct the path to the potential new split file
        # dynamic_split_filename = f"{extra_info['task']}_gta_{extra_info['variation']}_{extra_info['decomp']}_test_split.json"
        # dynamic_split_path = os.path.join(os.getcwd(), "spatial", "splits", dynamic_split_filename)
        
        # # Check if the dynamic split file exists
        # use_dynamic_split = os.path.exists(dynamic_split_path)
        # if use_dynamic_split:
        #     print(f"Found spatial split file. Using: {dynamic_split_path}")
        
        # # If it's the '0_00' noise level and the dynamic file exists, override the split path
        # if noise_level == '0_00' and use_dynamic_split:
        #     extra_info['split_path'] = dynamic_split_path
    
        image_path = f"/fast/AG_Kainmueller/data/GTA/{fold}/preprocessed/images/" 
        mask_path = f"/fast/AG_Kainmueller/data/GTA/{fold}/preprocessed/labels/" 
        uq_map_path = "/fast/AG_Kainmueller/data/GTA_CityScapes_UQ/"
        prediction_path = mask_path

        from datasets.GTA_CityScapes.gta_cityscapes_dataset_creation import OptimizedGTA_CityscapesDataset
        dataset = OptimizedGTA_CityscapesDataset(image_path, 
                                        mask_path, 
                                        uq_map_path, 
                                        prediction_path, 
                                        'abc',
                                        **extra_info)
        dataset.num_classes = 19 #previously 25, but strange that it doesn't result in 25
        
        # --- New Safeguard ---
        # Ensure all sample names are strings. This is crucial if the split file
        # contains numeric IDs that are parsed as integers.
        if dataset.sample_names and not isinstance(dataset.sample_names[0], str):
            print("Sample names are not strings. Converting to strings...")
            dataset.sample_names = [str(name) for name in dataset.sample_names]
        # --- End of Safeguard ---

        datasets[noise_level] = dataset
        print(f"Sample names for noise level {noise_level}:")
        print(dataset.sample_names[:5])  # Print first 5 sample names to verify        
    return datasets

def create_weedsgalore_datasets(uq_method):
    datasets = {}
    noise_levels = ['0_00', '1_00']
    
    for noise_level in noise_levels:
        extra_info = {
            'task' : 'crops_vs_weed',
            'variation': 'maize',
            'model_noise': 0,
            'data_noise': noise_level,
            'uq_method': uq_method,
            'decomp': 'pu',
            'spatial' : None,
            'metadata' : True,
            'split_path': None, 
            'split': None,
        }
        
        # # Construct the path to the potential new split file
        # dynamic_split_filename = f"{extra_info['task']}_weedsgalore_{extra_info['decomp']}_test_split.json"
        # dynamic_split_path = os.path.join(os.getcwd(), "spatial", "splits", dynamic_split_filename)
        
        # # Check if the dynamic split file exists
        # use_dynamic_split = os.path.exists(dynamic_split_path)
        # if use_dynamic_split:
        #     print(f"Found spatial split file. Using: {dynamic_split_path}")
        
        # # If it's the '0_00' noise level and the dynamic file exists, override the none split path
        # if noise_level == '0_00' and use_dynamic_split:
        #     extra_info['split_path'] = dynamic_split_path
    
        image_path = "/fast/AG_Kainmueller/data/weedsgalore/"
        uq_path =  "/fast/AG_Kainmueller/data/UQ_maps/weedsgalore/"

        from datasets.Weedsgalore.weedsgalore_dataset_creation import OptimizedWeedsGalore
        dataset = OptimizedWeedsGalore(image_path, 
                                        image_path, 
                                        uq_path, 
                                        uq_path, 
                                        'abc',
                                        **extra_info)
        dataset.num_classes = 3
        datasets[noise_level] = dataset
        
    return datasets

def create_lizard_datasets(task, uq_method):
    """Create Arctique datasets for both noise levels."""
    datasets = {}
    
    for noise_level in ['0_00', '1_00']:
        extra_info = {
            'task': task,
            'variation': 'glas_set',
            'model_noise': 0,
            'data_noise': noise_level,
            'uq_method': uq_method,
            'decomp': 'pu',
            'spatial': False,
            'metadata': False,
            'split_path': None, 
            'split': ['test'],
        }
        
        # # Construct the path to the potential new split file
        # dynamic_split_filename = f"{extra_info['task']}_lizard_{extra_info['variation']}_{extra_info['decomp']}_test_split.json"
        # dynamic_split_path = os.path.join(os.getcwd(), "spatial", "splits", dynamic_split_filename)
        
        # # Check if the dynamic split file exists
        # use_dynamic_split = os.path.exists(dynamic_split_path)
        # if use_dynamic_split:
        #     print(f"Found spatial split file. Using: {dynamic_split_path}")
        
        # # If it's the '0_00' noise level and the dynamic file exists, override the none split path
        # if noise_level == '0_00' and use_dynamic_split:
        #     extra_info['split_path'] = dynamic_split_path
        
        lmdb_path = '/fast/AG_Kainmueller/data/LizardRaw_new/archive/lizard_tiles.lmdb' 
        map_path = Path(f'/fast/AG_Kainmueller/data/Lizard_AggroUQ/trained_2/uncertainty_lizard_convnextv2_tiny_{extra_info["model_noise"]}')
        uq_map_path = map_path.joinpath('UQ_maps')
        prediction_path = map_path.joinpath('UQ_predictions')
        
        # Define split path to exlude tiles with exceeeding and wrong padding 
        json_path = Path(lmdb_path).parent.joinpath(f"/fast/AG_Kainmueller/data/LizardRaw_new/archive/lizard_dataset_splits_corrected.json")
        extra_info['split_path'] = json_path

        from datasets.Lizard.lizard_dataset_creation import LizardDataset
        
        dataset = LizardDataset(lmdb_path, 
                                lmdb_path, 
                                uq_map_path, 
                                prediction_path, 
                                'abc', 
                                **extra_info)
        dataset.num_classes = 7
        datasets[noise_level] = dataset
    return datasets

def create_wormbodies_datasets(variation, uq_method):
    """Create LIDC datasets for both noise levels."""
    datasets = {}
    noise_levels = ['0_00', '1_00']
    
    for noise_level in noise_levels:
        extra_info = {
            'task': 'fgbg',
            'variation': variation,
            'model_noise': 0,
            'data_noise': noise_level,
            'uq_method': uq_method,
            'decomp': 'pu',
            'spatial': None,
            'metadata': False,
            'split_path': None, 
            'split': ['test'],
        }
        
        # # Construct the path to the potential new split file
        # dynamic_split_filename = f"{extra_info['task']}_lidc_{extra_info['variation']}_{extra_info['decomp']}_test_split.json"
        # dynamic_split_path = os.path.join(os.getcwd(), "spatial", "splits", dynamic_split_filename)
        
        # # Check if the dynamic split file exists
        # use_dynamic_split = os.path.exists(dynamic_split_path)
        # if use_dynamic_split:
        #     print(f"Found spatial split file. Using: {dynamic_split_path}")
        
        # # If it's the '0_00' noise level and the dynamic file exists, override the none split path
        # if noise_level == '0_00' and use_dynamic_split:
        #     extra_info['split_path'] = dynamic_split_path
        
        data_path = '/fast/AG_Kainmueller/data/'
        uq_map_path = '/fast/AG_Kainmueller/data/UQ_maps/wormbodies/'
        
        from datasets.Wormbodies.wormbodies_dataset_creation import wormbodies_dataset
        
        dataset = wormbodies_dataset(data_path, data_path, uq_map_path, uq_map_path, 'abc', **extra_info)
        dataset.num_classes = 2
        datasets[noise_level] = dataset
    
    return datasets

def evaluate_correlation_across_datasets(all_datasets_dict, sample_size, num_workers, output_name="cross_dataset"):
    """
    Evaluate correlations across multiple datasets by concatenating all data.
    
    Args:
        all_datasets_dict: Dictionary with dataset names as keys and dataset dictionaries as values
                          Format: {'ade20k': {'0_00': dataset1, '1_00': dataset2}, 'gta': {...}}
        sample_size: Number of samples per dataset
        num_workers: Number of parallel workers
        output_name: Base name for output files
    """
    all_summary_dfs = []
    
    # Process each dataset and its noise levels
    for dataset_name, datasets_dict in all_datasets_dict.items():
        print(f"\n=== Processing {dataset_name} ===")
        
        for noise_level, dataset in datasets_dict.items():
            full_dataset_name = f"{dataset_name}_{noise_level}"
            summary_df = process_single_dataset(dataset, full_dataset_name, sample_size, num_workers)
            
            # Add dataset and noise level columns
            summary_df.insert(loc=1, column="base_dataset", value=dataset_name)
            summary_df.insert(loc=2, column="noise_level", value=noise_level)
            all_summary_dfs.append(summary_df)
    
    # Concatenate all datasets
    combined_df = pd.concat(all_summary_dfs, ignore_index=True)
    
    # Save combined summary
    out_name = f"aggregation_value_summary_joint_noise_{output_name}"
    combined_df.to_csv(os.path.join("output", "tables", f"{out_name}.csv"), index=False) #"joint_correlation",
    print(f"\nCombined aggregation value summary {out_name}.csv saved to output folder.")
    
    # # Compute correlations between methods (columns), not samples (rows)
    # method_columns = [col for col in combined_df.columns if col not in ['uq_map_name', 'dataset_name', 'base_dataset', 'noise_level']]
    
    # # Use only the method columns for correlation computation
    # correlation_df = combined_df[method_columns]
    
    # start = time.time()
    # correlations = {}
    # for correlation_type in ["pearson", "spearman", "kendall"]:
    #     # Compute correlation between methods (columns)
    #     corr_matrix = correlation_df.corr(method=correlation_type, min_periods=1)
    #     correlations[correlation_type] = corr_matrix
    # print(f"Computed correlation matrices: {time.time() - start} s")
    
    # # Save correlation matrices and plots
    # for correlation_type, corr_matrix in correlations.items():
    #     out_name = f"correlation_matrix_{correlation_type}_joint_noise_{output_name}"
        
    #     # Save to csv
    #     corr_matrix.to_csv(os.path.join("output", "tables", "joint_correlation", f"{out_name}.csv"))
    #     print(f"Correlation matrix {out_name}.csv saved to output folder.")
        
    #     # Save heatmap as png
    #     save_correlation_matrix_plot(corr_matrix, out_name, correlation_type, os.path.join("output", "figures", "joint_correlation"))
    #     print(f"Correlation heatmap {out_name}.png saved to output folder.")
    
    # return combined_df, correlations

def run_cross_dataset_analysis(uq_method, sample_size=0, num_workers=16):
    """
    Run correlation analysis across all datasets.
    
    Args:
        uq_method: UQ method to use ('dropout', 'softmax', etc.)
        sample_size: Number of samples per dataset (0 for all)
        num_workers: Number of parallel workers
    """
    # Create output directories
    os.makedirs("output/tables/joint_correlation", exist_ok=True)
    os.makedirs("output/figures/joint_correlation", exist_ok=True)
    
    # Initialize dictionary to store all datasets
    all_datasets = {}
    
    # ADE20K
    print("Creating ADE20K datasets...")
    model_names = ['deeplabv3']
    model_ids = ['deeplabv3_r50-d8_4xb4-160k_ade20k-512x512']
    for model_name, model_id in zip(model_names, model_ids):
        datasets = create_ade20k_datasets(model_id, uq_method)
        all_datasets[f'ade20k_{model_name}'] = datasets
    
    # Arctique
    print("Creating Arctique datasets...")
    for task in ['semantic', 'instance']: 
        datasets = create_arctique_datasets(task, uq_method)
        variation = 'blood_cells' if task == 'semantic' else 'nuclei_intensity'
        all_datasets[f'arctique_{task}_{variation}'] = datasets
    
    # LIDC
    print("Creating LIDC datasets...")
    for variation in ['malignancy', 'texture']: #'malignancy'
        datasets = create_lidc_datasets(variation, uq_method)
        all_datasets[f'lidc_{variation}'] = datasets
    
    # GTA
    print("Creating GTA datasets...")
    datasets = create_gta_datasets(uq_method)
    all_datasets['gta'] = datasets
    
    # Weedsgalore
    print("Creating Weedsgalore datasets...")
    datasets = create_weedsgalore_datasets(uq_method)
    all_datasets['weedsgalore'] = datasets
    
    # Lizard
    print("Creating Lizard datasets...")
    datasets = create_lizard_datasets(uq_method)
    all_datasets['lizard'] = datasets
    
    # Wormbodies
    print("Creating Wormbodies datasets...")
    datasets = create_wormbodies_datasets(uq_method)
    all_datasets['wormbodies'] = datasets
    
    # Run cross-dataset analysis
    print(f"\n=== Running cross-dataset correlation analysis ===")
    output_name = f"all_datasets_{uq_method}_pu"
    combined_df, correlations = evaluate_correlation_across_datasets(
        all_datasets, sample_size, num_workers, output_name
    )
    
    # Print summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"Total samples across all datasets: {len(combined_df)}")
    print(f"Datasets included: {list(all_datasets.keys())}")
    print(f"Method columns: {len([col for col in combined_df.columns if col not in ['uq_map_name', 'dataset_name', 'base_dataset', 'noise_level']])}")
    
    # Print dataset breakdown
    dataset_counts = combined_df['base_dataset'].value_counts()
    print(f"\nSamples per dataset:")
    for dataset, count in dataset_counts.items():
        print(f"  {dataset}: {count}")
    
    return combined_df, correlations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create correlation matrix across noise levels')
    parser.add_argument('--dataset', type=str, help='Dataset name: ade20k, arctique, lidc, etc. or "all" for cross-dataset analysis')
    parser.add_argument('--uq_method', type=str, help='UQ method: dropout, softmax')
    parser.add_argument('--sample_size', type=int, default=0, help='Number of samples per dataset')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of parallel workers')
    parser.add_argument('--compute_individual', type=bool, default=False, help='Compute individual noise level correlations')
    args = parser.parse_args()

    # Create output directories
    os.makedirs("output/tables/joint_correlation", exist_ok=True)
    os.makedirs("output/figures/joint_correlation", exist_ok=True)

    if args.dataset == "all":
        # Cross-dataset analysis
        combined_df, correlations = run_cross_dataset_analysis(args.uq_method, args.sample_size, args.num_workers)
    
    if args.dataset == "ade20k":
        model_names = ['deeplabv3']
        model_ids = ['deeplabv3_r50-d8_4xb4-160k_ade20k-512x512']
        for model_name, model_id in zip(model_names, model_ids):
            datasets = create_ade20k_datasets(model_id, args.uq_method)
            base_name = f"joint_noise_ade20k_{model_name}_semantic_cityscapes_{args.uq_method}_pu"
            evaluate_correlation_across_noise_levels(datasets, args.sample_size, args.num_workers, base_name, args.compute_individual)

    elif args.dataset == "arctique":
        for task in ['instance', 'semantic']:
            datasets = create_arctique_datasets(task, args.uq_method)
            variation = 'blood_cells' if task == 'semantic' else 'nuclei_intensity'
            base_name = f"joint_noise_arctique_{task}_{variation}_{args.uq_method}_pu"
            args.num_workers = 8
            evaluate_correlation_across_noise_levels(datasets, args.sample_size, args.num_workers, base_name, args.compute_individual)

    elif args.dataset == "lidc":
        for variation in ['malignancy', 'texture']:
            datasets = create_lidc_datasets(variation, args.uq_method)
            base_name = f"joint_noise_lidc_fgbg_{variation}_{args.uq_method}_pu"
            evaluate_correlation_across_noise_levels(datasets, args.sample_size, args.num_workers, base_name, args.compute_individual)
    
    elif args.dataset == "gta":
        datasets = create_gta_datasets(args.uq_method)
        base_name = f"joint_noise_gta_semantic_cityscapes_{args.uq_method}_pu"
        evaluate_correlation_across_noise_levels(datasets, args.sample_size, args.num_workers, base_name, args.compute_individual)
    
    elif args.dataset == "weedsgalore":
        datasets = create_weedsgalore_datasets(args.uq_method)
        base_name = f"joint_noise_weedsgalore_crops_vs_weed_maize_{args.uq_method}_pu"
        evaluate_correlation_across_noise_levels(datasets, args.sample_size, args.num_workers, base_name, args.compute_individual)
    
    elif args.dataset == "lizard":
        for task in ['instance', 'semantic']:
            datasets = create_lizard_datasets(task, args.uq_method)
            base_name = f"joint_noise_lizard_{task}_glas_set_{args.uq_method}_pu"
            args.num_workers = 2
            evaluate_correlation_across_noise_levels(datasets, args.sample_size, args.num_workers, base_name, args.compute_individual)
    
    elif args.dataset == "wormbodies":
        for variation in ['nematodes', 'protists']:
            datasets = create_wormbodies_datasets(variation, args.uq_method)
            base_name = f"joint_noise_wormbodies_fgbg_{variation}_{args.uq_method}_pu"
            evaluate_correlation_across_noise_levels(datasets, args.sample_size, args.num_workers, base_name, args.compute_individual)
        
    # Add other datasets as needed (weedsgalore, lizard...)
    # following the same pattern...