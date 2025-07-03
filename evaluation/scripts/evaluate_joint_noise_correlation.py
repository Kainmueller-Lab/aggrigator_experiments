import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import multiprocessing
import time
import yaml
import argparse

from pathlib import Path
from joblib import Parallel, delayed

from aggrigator.uncertainty_maps import UncertaintyMap
from aggrigator.methods import AggregationMethods as am
from aggrigator.summary import AggregationSummary


focus_strategy_list = [
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
    #(am.above_quantile_mean, 0.95),
    (am.above_quantile_mean_fg_ratio, None),
    (am.patch_aggregation, 10), 
    (am.patch_aggregation, 20),
    (am.patch_aggregation, 40),
    (am.patch_aggregation, 80),
    (am.patch_aggregation, 100),
    #(am.patch_aggregation, 200),
    (am.class_mean_w_equal_weights, None),
    (am.class_mean_weighted_by_occurrence, None),
]


def load_dataset_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_correlation_matrix_plot(df, filename, save_dir):
    """
    Computes and plots the correlation matrix of methods across columns.
    """
    # Compute the correlation matrix (rows as methods, columns as features)
    corr_matrix = df[df.columns.tolist()[1:]].T.corr(min_periods=1)

    # Plot the correlation matrix as a heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    strategy_names = df.index.tolist()
    sns.heatmap(corr_matrix, ax=ax, cmap="inferno", annot=False, fmt=".2f",
                cbar=True, vmin=-1, vmax=1, xticklabels=strategy_names, yticklabels=strategy_names)
    
    # Color strategy names by category
    color_code = {
        "threshold": (208, 247, 227),
        "quantile": (99, 72, 97),
        "patch": (106, 146, 184),
        "class_mean": (197, 159, 214),
    }
    for tick in ax.get_xticklabels():
        strategy_name = tick.get_text()
        color = next((color_code[key] for key in color_code if key in strategy_name), "black")
        tick.set_bbox(dict(facecolor=color, edgecolor='none', alpha=0.5, boxstyle="round,pad=0.3"))
    for tick in ax.get_yticklabels():
        strategy_name = tick.get_text()
        color = next((color_code[key] for key in color_code if key in strategy_name), "black")
        tick.set_bbox(dict(facecolor=color, edgecolor='none', alpha=0.5, boxstyle="round,pad=0.3"))

    plt.title(filename)
    plt.savefig(os.path.join(save_dir, f"{filename}.png"))
    plt.close()


def compute_correlations(df):
    method_columns = df.columns.tolist()[1:]
    correlations = {}
    for correlation_type in ["pearson", "spearman", "kendall"]:
        corr_matrix = df[method_columns].T.corr(min_periods=1, method=correlation_type)
        corr_matrix.columns = [strat for strat in df["Name"].tolist()]
        corr_matrix.index = [strat for strat in df["Name"].tolist()]
        correlations[correlation_type] = corr_matrix
    return correlations


def process_single_dataset(dataset, dataset_name, sample_size, num_workers):
    """
    Process a single dataset and return the summary dataframe.
    """
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
        if dataset.num_classes != 2 and (am.above_quantile_mean_fg_ratio, None) in focus_strategy_list:
            focus_strategy_list.remove((am.above_quantile_mean_fg_ratio, None))
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
        patch_200_in_agg_list = (am.patch_aggregation, 200) in focus_strategy_list
        if patch_200_in_agg_list and (h < 200 or w < 200):
            print(f"Warning: Ignoring UQ map {sample['sample_name']} because it is too small.")
            return None
        
        # Normalize arrays by ln(K) where K is number of classes
        if dataset_info['num_classes'] is not None:
            uq_array = uq_array / np.log(dataset_info['num_classes'])

        # Apply aggregation strategies
        uq_map = UncertaintyMap(array=uq_array, mask=prediction, name=sample['sample_name'])
        summary = AggregationSummary(focus_strategy_list, num_cpus=1)
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


def evaluate_correlation_across_noise_levels(datasets_dict, sample_size, num_workers, base_dataset_name):
    """
    Evaluate correlations across different noise levels by concatenating data.
    
    Args:
        datasets_dict: Dictionary with noise levels as keys and datasets as values
        sample_size: Number of samples per dataset
        num_workers: Number of parallel workers
        base_dataset_name: Base name for the dataset
    """
    all_summary_dfs = []
    
    # Process each dataset (noise level)
    for noise_level, dataset in datasets_dict.items():
        dataset_name = f"{base_dataset_name}_{noise_level}"
        summary_df = process_single_dataset(dataset, dataset_name, sample_size, num_workers)
        
        # Add noise level column
        summary_df.insert(loc=2, column="noise_level", value=noise_level)
        all_summary_dfs.append(summary_df)
    
    # Concatenate all datasets
    combined_df = pd.concat(all_summary_dfs, ignore_index=True)
    
    # Save combined summary
    out_name = f"aggregation_value_summary_{base_dataset_name}_combined"
    combined_df.to_csv(os.path.join("output", "tables", f"{out_name}.csv"), index=False)
    print(f"Combined aggregation value summary {out_name}.csv saved to output folder.")
    
    # Compute correlations on combined data
    method_columns = [col for col in combined_df.columns if col not in ['uq_map_name', 'dataset_name', 'noise_level']]
    correlation_df = combined_df[['uq_map_name'] + method_columns].set_index('uq_map_name')
    
    start = time.time()
    correlations = {}
    for correlation_type in ["pearson", "spearman", "kendall"]:
        corr_matrix = correlation_df.T.corr(min_periods=1, method=correlation_type)
        correlations[correlation_type] = corr_matrix
    print(f"Computed correlation matrices: {time.time() - start} s")
    
    # Save correlation matrices and plots
    for correlation_type, corr_matrix in correlations.items():
        out_name = f"correlation_matrix_{correlation_type}_{base_dataset_name}_combined"
        
        # Save to csv
        corr_matrix.to_csv(os.path.join("output", "tables", f"{out_name}.csv"))
        print(f"Correlation matrix {out_name}.csv saved to output folder.")
        
        # Save heatmap as png
        save_correlation_matrix_plot(corr_matrix, out_name, os.path.join("output", "figures"))
        print(f"Correlation heatmap {out_name}.png saved to output folder.")


# Modified dataset creation functions
def create_ade20k_datasets(model_name, uq_method):
    """Create ADE20K datasets for both noise levels."""
    datasets = {}
    noise_levels = ['0_00', '1_00']
    folders = ['validation', 'test_cityscapes']
    
    for nl, fold in zip(noise_levels, folders):
        extra_info = {
            'task': 'semantic',
            'variation': 'cityscapes',
            'model_noise': 0,
            'data_noise': nl,
            'uq_method': uq_method,
            'decomp': 'pu',
            'spatial': None,
            'split_path': None,
            'split': None,
            'metadata': False,
            'model_checkpoint': model_name,
        }
        
        image_path = f'/fast/AG_Kainmueller/data/ADEChallengeData2016/images/{fold}'
        mask_path = f'/fast/AG_Kainmueller/data/ADEChallengeData2016/annotations/{fold}'
        uq_map_path = f'/fast/AG_Kainmueller/data/UQ_maps/ADE20K/'
        prediction_path = '/fast/AG_Kainmueller/data/ADEChallengeData2016/'
        
        from datasets.ADE20K.ade20k_dataset_creation import ADE20K_CityscapesDataset
        dataset = ADE20K_CityscapesDataset(image_path, mask_path, uq_map_path, prediction_path, 
                                          '/fast/AG_Kainmueller/data/ADEChallengeData2016/objectInfo150.json',
                                          **extra_info)
        dataset.num_classes = 150
        datasets[nl] = dataset
    
    return datasets


def create_arctique_datasets(task, uq_method):
    """Create Arctique datasets for both noise levels."""
    datasets = {}
    noise_levels = ['0_00', '1_00']
    
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
            'metadata': False,
        }
        
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
    global focus_strategy_list
    if (am.patch_aggregation, 200) in focus_strategy_list:
        focus_strategy_list.remove((am.patch_aggregation, 200))
    
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
            'metadata': True,
            'render_2d': True,
            'render_ind_masks': False,
        }
        
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
    datasets = {}
    noise_levels = ['0_00', '1_00']
    split_path = ["/fast/AG_Kainmueller/data/GTA_ValUES_splits/GTA_id_test", None]
    folders = ['OriginalData', 'CityScapesOriginalData']
    
    # Remove patch aggregation 200 for LIDC
    global focus_strategy_list
    
    for noise_level, split, fold in zip(noise_levels, split_path, folders):
        extra_info = {
            'task': 'semantic',
            'variation': 'cityscapes',
            'model_noise': 0,
            'data_noise': noise_level,
            'uq_method': uq_method,
            'decomp': 'pu',
            'spatial' : None,
            'split_path' : split, 
            'split' : None
        }
    
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
        dataset.num_classes = 32
        datasets[noise_level] = dataset
        
    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create correlation matrix across noise levels')
    parser.add_argument('--dataset', type=str, help='Dataset name: ade20k, arctique, lidc, etc.')
    parser.add_argument('--uq_method', type=str, help='UQ method: dropout, softmax')
    parser.add_argument('--sample_size', type=int, default=0, help='Number of samples per dataset')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of parallel workers')
    args = parser.parse_args()

    # Create output directories
    os.makedirs("output/tables", exist_ok=True)
    os.makedirs("output/figures", exist_ok=True)

    if args.dataset == "ade20k":
        for model_name in ['deeplabv3']:
            datasets = create_ade20k_datasets(model_name, args.uq_method)
            base_name = f"joint_noise_ade20k_{model_name}_semantic_cityscapes_{args.uq_method}_pu"
            evaluate_correlation_across_noise_levels(datasets, args.sample_size, args.num_workers, base_name)

    elif args.dataset == "arctique":
        for task in ['instance', 'semantic']:
            datasets = create_arctique_datasets(task, args.uq_method)
            variation = 'blood_cells' if task == 'semantic' else 'nuclei_intensity'
            base_name = f"joint_noise_arctique_{task}_{variation}_{args.uq_method}_pu"
            evaluate_correlation_across_noise_levels(datasets, args.sample_size, args.num_workers, base_name)

    elif args.dataset == "lidc":
        for variation in ['malignancy', 'texture']:
            datasets = create_lidc_datasets(variation, args.uq_method)
            base_name = f"joint_noise_lidc_fgbg_{variation}_{args.uq_method}_pu"
            evaluate_correlation_across_noise_levels(datasets, args.sample_size, args.num_workers, base_name)
    
    elif args.dataset == "gta":
        datasets = create_gta_datasets(args.uq_method)
        base_name = f"joint_noise_gta_semantic_cityscapes_{args.uq_method}_pu"
        evaluate_correlation_across_noise_levels(datasets, args.sample_size, args.num_workers, base_name)
        

    # Add other datasets as needed (weedsgalore, lizard...)
    # following the same pattern...