import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import multiprocessing
from joblib import Parallel, delayed
import time
import yaml

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
    (am.above_quantile_mean, 0.95),
    (am.patch_aggregation, 10), 
    (am.patch_aggregation, 20),
    (am.patch_aggregation, 40),
    (am.patch_aggregation, 80),
    (am.patch_aggregation, 100),
    (am.patch_aggregation, 200),
]


# TODO: Move to a better utils file.
def load_dataset_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_correlation_matrix_plot(df, filename, save_dir):
    """
    Computes and plots the correlation matrix of methods across columns.

    :param df: pandas DataFrame where each row represents a method and columns represent features.
    """
    # Compute the correlation matrix (rows as methods, columns as features)
    corr_matrix = df[df.columns.tolist()[1:]].T.corr(min_periods=1)

    # Plot the correlation matrix as a heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    strategy_names = df.index.tolist()
    sns.heatmap(corr_matrix, ax=ax, cmap="coolwarm", annot=False, fmt=".2f",
                cbar=True, vmin=-1, vmax=1, xticklabels=strategy_names, yticklabels=strategy_names)
    
    # Color strategy names by category
    color_code = {
        "threshold": "red",
        "quantile": "green",
        "patch": "blue"
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


# def to_correlation_matrix(df):
#     method_columns = df.columns.tolist()[1:]
#     corr_matrix = df[method_columns].T.corr(min_periods=1)
#     # Change index and columns names to method_columns
#     corr_matrix.columns = [strat for strat in df["Name"].tolist()]
#     corr_matrix.index = [strat for strat in df["Name"].tolist()]
#     return corr_matrix

def compute_correlations(df):
    method_columns = df.columns.tolist()[1:]
    correlations = {}
    for correlation_type in ["pearson", "spearman", "kendall"]:
        corr_matrix = df[method_columns].T.corr(min_periods=1, method=correlation_type)
        corr_matrix.columns = [strat for strat in df["Name"].tolist()]
        corr_matrix.index = [strat for strat in df["Name"].tolist()]
        correlations[correlation_type] = corr_matrix
    return correlations



def evaluate_correlation(dataset, sample_size, num_workers):
    sample_size = len(dataset) if sample_size == 0 else sample_size

    # Print info
    dataset_info = dataset.get_info()
    dataset_info.pop('semantic_mapping') # NOTE: Semantic mapping too long in case of many classes
    print("____________________")
    print(f"Evaluating correlation matrix")
    for key, value in dataset_info.items():
        print(f"{key}: {value}")
    print(f"Number of samples used for correlation matrix: {sample_size} of {len(dataset)}")
    if dataset_info['num_classes'] is not None:
        print(f"NOTE: Normalizing UQ maps by ln(K) where K={dataset_info['num_classes']} is the number of classes.")
    else:
        print(f"NOTE: Could not normalize UQ maps because dataset_info['num_classes'] is not defined.")
    print("____________________")


    def aggregate(sample):
        # Load uncertainty maps and masks from dataset
        mask = sample['mask']
        uq_array = sample['uq_map']

        # Slice if 3D
        if uq_array.ndim == 3:
            print(f"Warning: 3D UQ map detected. Only 2D slices are used for correlation matrix.")
            mid_slice = uq_array.shape[0] // 2
            uq_array = uq_array[mid_slice, :, :]
            mask = mask[mid_slice, :, :]

        # Ignore too small images bc of patch aggregation with patch size 200
        h, w = uq_array.shape
        if h < 200 or w < 200:
            print(f"Warning: UQ map {sample['sample_name']} is too small for patch aggregation with patch size 200.")
            return None
        
        # Replace negative values with zero
        # NOTE: Such values (close to zero) sometimes occur and need to be dealt with.
        uq_array = np.where(uq_array < 0, 0, uq_array)
        
        # Normalize arrays by ln(K) where K is number of classes if UQ maps are not normalized in dataloader
        if dataset_info['num_classes'] is not None:
            uq_array = uq_array / np.log(dataset_info['num_classes'])

        # Apply aggregation strategies
        uq_map = UncertaintyMap(array=uq_array, mask=mask, name=sample['sample_name'])
        summary = AggregationSummary(focus_strategy_list, num_cpus=1)
        return summary.apply_methods([uq_map], save_to_excel=False, do_plot=False, max_value=1.0)
    
    # Aggregate all UQ maps
    start = time.time()
    n_jobs = multiprocessing.cpu_count() if num_workers == 0 else num_workers
    summary_dfs = Parallel(n_jobs=n_jobs, verbose=10)(delayed(aggregate)(dataset[idx]) for idx in range(sample_size))
    summary_dfs = [df.set_index("Name") for df in summary_dfs if df is not None]
    summary_df = pd.concat(summary_dfs, axis=1).reset_index()
    print(f"Computed aggregation strategy summary: {time.time() - start} s")

    # Compute the correlation matrices: Pearson, Spearman, Kendall
    start = time.time()
    correlations = compute_correlations(summary_df)
    print(f"Computed correlation matrices: {time.time() - start} s")

    try:
        dataset_name = dataset.get_info()['dataset_name'] # TODO: Find a better default identifier.
    except:
        dataset_name = "" # NOTE: Please add dataset_name member to dataset class
        
    for correlation_type, corr_matrix in correlations.items():
        out_name = f"correlation_matrix_{correlation_type}_{dataset_name}"

        # Save to csv
        corr_matrix.to_csv(os.path.join("output", "tables", f"{out_name }.csv"))
        print(f"Correlation matrix {out_name}.csv saved to output folder.")

        # Save heatmap as png
        save_correlation_matrix_plot(corr_matrix, out_name, os.path.join("output", "figures"))
        print(f"Correlation heatmap {out_name}.png saved to output folder.")



import argparse

from evaluation.scripts.evaluate_correlation import evaluate_correlation, load_dataset_config
from datasets.ADE20K.ade20k_loader import ADE20K


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create correlation matrix for aggregation strategies evaluated on a dataset')
    parser.add_argument('--dataset_config', type=str, default='configs/ade20k_deeplabv3.yaml', help='Path to config file')
    parser.add_argument('--sample_size', type=int, default='0', help='Number of samples from dataset used to evaluate correlation matrix. If 0, all samples are used.')
    parser.add_argument('--num_workers', type=int, default='8', help='Number of workers for parallel processing. If 0, all available CPUs are used.')
    args = parser.parse_args()
    
    config = load_dataset_config(args.dataset_config)
    
    dataset = ADE20K(config['image_dir'],
                    config['label_dir'],
                    config['uq_map_dir'],
                    config['prediction_dir'],
                    config['metadata_dir'])
    
    evaluate_correlation(dataset, args.sample_size, args.num_workers)