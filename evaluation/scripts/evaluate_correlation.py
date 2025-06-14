import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import multiprocessing
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

# Filter Lizard and Arctique by UQ method and task
UQ_METHOD = "pu"
TASK = "semantic"
LIZARD_DIR = "lizard_unnormalized_noncal_uq_maps"
ARCTIQUE_DIR = "arctique_unnormalized_noncal_uq_maps"
CROPS_DIR = "crops_uq_maps"
NOISE_LEVEL = "0_00"


def norm_factor(file, uq_method, task):
    if "softmax" in file:
        return 1
    if task == "semantic":
        return 1 / np.log(7)
    else:
        return 1 / np.log(3) # 3-class FG-BG-Segmentation
    

def get_lizard_arrays(dir, uq_method="pu", task="semantic", uq_model="dropout"):
    # Load arrays
    path = os.path.join(os.getcwd(), "..", "..", "data", dir)
    files = os.listdir(path)
    # Filter files by UQ method and segmentation task
    def norm_factor(file, task):
        if "softmax" in file:
            return 1
        return 1 / np.log(7) if task == "semantic" else 1 / np.log(3)
    arrays = [np.load(os.path.join(path, file))*norm_factor(file, task) for file in files if uq_method in file and task in file and uq_model in file] # 3D arrays, shape (50,512,512)
    num_slices = arrays[0].shape[0]
    arrays = [array[slice,:,:] for array in arrays for slice in range(num_slices)]
    arrays = [np.where(array < 0, 0, array) for array in arrays]
    # # Uniformly scale by ln(K) where K is number of classes
    # # 7 for Lizard: Background, Neutrophils, LYM, PLA, EPI, EOS, FIB 
    # if task == "semantic":
    #     arrays /= np.log(7)
    # else:
    #     arrays /= np.log(3) # 3-class FG-BG-Segmentation
    files = [f"{file}_{slice}" for file in files for slice in range(num_slices)]
    return arrays, files

def get_arctique_arrays(dir, uq_method="pu", task="semantic", noise_level="0_00", uq_model="dropout"):
    # Load arrays
    path = os.path.join(os.getcwd(), "..", "..", "data", dir)
    files = os.listdir(path)
    # Filter files by UQ method and segmentation task and noise level
    def norm_factor(file, task):
        if "softmax" in file:
            return 1
        return 1 / np.log(6) if task == "semantic" else 1 / np.log(3)
    arrays = [np.load(os.path.join(path, file))*norm_factor(file, task) for file in files if uq_method in file and task in file and noise_level in file and uq_model in file] # 3D arrays, shape (50,512,512)
    num_slices = arrays[0].shape[0]
    arrays = [array[slice,:,:] for array in arrays for slice in range(num_slices)]
    arrays = [np.where(array < 0, 0, array) for array in arrays]
    # # Uniformly scale by ln(K) where K is number of classes
    # # 6 for Arctique: Background, LYM, PLA, EPI, EOS, FIB 
    # if task == "semantic":
    #     arrays /= np.log(6)
    # else:
    #     arrays /= np.log(3)
    files = [f"{file}_{slice}" for file in files for slice in range(num_slices)]
    return arrays, files

def get_crops_arrays(dir):
    path = os.path.join(os.getcwd(), "..", "..", "data", dir)
    files = os.listdir(path)
    arrays = [np.load(os.path.join(path, file)) for file in files] # 2D arrays
    arrays = [np.where(array < 0, 0, array) for array in arrays]
    return arrays, files














def save_correlation_matrix_plot(df, filename, save_dir):
    """
    Computes and plots the correlation matrix of methods across columns.

    :param df: pandas DataFrame where each row represents a method and columns represent features.
    """
    # Compute the correlation matrix (rows as methods, columns as features)
    corr_matrix = df[df.columns.tolist()[1:]].T.corr(min_periods=1)

    # Plot the correlation matrix as a heatmap
    plt.figure(figsize=(8, 6))
    #sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, 
                xticklabels=df.columns, yticklabels=df.index, vmin=-1, vmax=1)  # Show row names

    plt.title(filename)
    plt.savefig(os.path.join(save_dir, f"{filename}.png"))
    plt.close()


def load_dataset_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def to_correlation_matrix(df):
    method_columns = df.columns.tolist()[1:]
    corr_matrix = df[method_columns].T.corr(min_periods=1)
    # Change index and columns names to method_columns
    corr_matrix.columns = [strat for strat in df["Name"].tolist()]
    corr_matrix.index = [strat for strat in df["Name"].tolist()]
    return corr_matrix


def max_num_cpus():
    return multiprocessing.cpu_count()


def load_unc_data(dataset, sample_size):
    uq_arrays = []
    masks = []
    filenames = []

    samples = [dataset[i] for i in range(min(sample_size, len(dataset)))]

    count = 0
    for sample in dataset:
        if sample_size > 0 and count >= sample_size:
            break

        uq = sample['uq_map']
        mask = sample['mask']
        name = sample['sample_name']

        # Slice if 3D
        if uq.ndim == 3:
            mid_slice = uq.shape[0] // 2
            uq = uq[mid_slice, :, :]
            mask = mask[mid_slice, :, :]

        h, w = uq.shape
        if h >= 200 and w >= 200:
            uq_arrays.append(uq)
            masks.append(mask)
            filenames.append(name)
            count += 1  # Only count valid samples

        # TODO: Normalize arrays by ln(K) where K is number of classes

    return uq_arrays, masks, filenames


def evaluate_correlation(dataset, sample_size):
    # Print info
    dataset_info = dataset.get_info()
    dataset_info.pop('semantic_mapping') # NOTE: Semantic mapping too long bc of 150 classes
    print("____________________")
    print(f"Evaluating correlation matrix")
    for key, value in dataset_info.items():
        print(f"{key}: {value}")
    print(f"Number of samples used for correlation matrix: {sample_size} of {len(dataset)}")
    print("____________________")

    # Load uncertainty maps and masks from dataset
    # TODO: Add random sampling an seed
    start = time.time()
    arrays, masks, filenames= load_unc_data(dataset, sample_size)
    print(f"Loaded uncertainty maps and masks from dataset: {time.time() - start} s")

    # Create UncertaintyMap summary
    # TODO: Maybe batch bc not all 2000 arrays + masks should be loaded at once into the summary...
    # TODO: Solve empty values in correlation amtrix for high thresholds
    start = time.time()
    uq_maps = [UncertaintyMap(array=array, mask=mask, name=name) for array, mask, name in zip(arrays, masks, filenames)]
    summary = AggregationSummary(focus_strategy_list, name=f"{dataset_info['dataset_name']}_{dataset_info['model_name']}_summary", num_cpus=max_num_cpus())
    summary_df = summary.apply_methods(uq_maps, save_to_excel=False, do_plot=False, max_value=1.0)
    print(f"Computed aggregation strategy summary: {time.time() - start} s")

    # Compute the correlation matrix
    start = time.time()
    corr_matrix = to_correlation_matrix(summary_df)
    out_name = f"correlation_matrix_{dataset_info['dataset_name']}_{dataset_info['model_name']}"
    print(f"Computed correlation matrix: {time.time() - start} s")

    # Save to csv
    corr_matrix.to_csv(os.path.join("output", "tables", f"{out_name }.csv"))
    print(f"Correlation matrix {out_name}.csv saved to output folder.")

    # Save heatmap as png
    save_correlation_matrix_plot(corr_matrix, out_name, os.path.join("output", "figures"))
    print(f"Correlation heatmap {out_name}.png saved to output folder.")





from datasets.ADE20K.ade20k_loader import ADE20K

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create correlation matrix for aggregation strategies evaluated on a dataset')
    parser.add_argument('--dataset_config', type=str, default='configs/ade20k_deeplabv3.yaml', help='Path to config file')
    parser.add_argument('--sample_size', type=int, default='200', help='Number of samples from dataset used to evaluate correlation matrix')
    args = parser.parse_args()
    
    config = load_dataset_config(args.dataset_config)
    
    dataset = ADE20K(config['image_dir'],
                    config['label_dir'],
                    config['uq_map_dir'],
                    config['prediction_dir'],
                    config['metadata_dir'])
    
    evaluate_correlation(dataset, args.sample_size)

    

