import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import multiprocessing
import yaml

from aggrigator.uncertainty_maps import UncertaintyMap
from aggrigator.methods import AggregationMethods as am
from aggrigator.summary import AggregationSummary


dataset_names = ["lizard", "arctique", "weedsgalore", "ade20k", "licd"]

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


def load_unc_data_old(config, uq_method, unc_measure):
    pass



def load_unc_data(config, uq_method, unc_measure):
    uq_map_dir = config['uq_map_dir'] # TODO: Adapt to config sturcture.
    gt_label_dir = config['label_dir']
    uq_map_files = os.listdir(uq_map_dir)

    # Filter filenames
    is_valid_file = lambda filename: filename.endswith(".npy") and uq_method in filename and unc_measure in filename # TODO: Add task filter?
    filenames = [file for file in uq_map_files if is_valid_file(file)]
    filenames = filenames[:200]

    # Deal with 3D arrays
    if config['map_dimension'] == 3:
        # TODO: Warn and use 2D slice of 3D arrays
        pass

    arrays_dict = {}
    masks_dict = {}
    for file in filenames:
        # Load arrays
        path = os.path.join(uq_map_dir, file)
        arr = np.load(path, allow_pickle=True)
        if arr.shape[0] < 200 or arr.shape[1] < 200: # NOTE: ADE20K has few files with width <200, leading to errors with PatchAggregation and patch size 200.
            # TODO: Find better solution for patch size if image too small
            print(f"Warning: Array {file} has shape {arr.shape} < 200. It will be ignored since this leads to errors with PatchAggregation of size 200x200.")
            continue
        arrays_dict[file] = arr

        # Load masks
        base_name = file.split("_")[0]
        mask_path = os.path.join(gt_label_dir, base_name + ".npy") # TODO: Check this
        try:
            mask = np.load(mask_path, allow_pickle=True)
            masks_dict[file] = mask
        except:
            masks_dict[file] = None

    final_filenames = arrays_dict.keys()
    arrays = arrays_dict.values()
    # TODO: Normalize arrays if not done already
    arrays = [np.where(array < 0, 0, array) for array in arrays] # TODO: Write warnings, but often some -0.000 values may appear
    masks = masks_dict.values()

    assert len(masks) == len(arrays), f"Number of masks ({len(masks)}) does not match number of uncertainty maps ({len(arrays)})"
    assert len(final_filenames) == len(arrays), f"Number of filenames ({len(final_filenames)}) does not match number of uncertainty maps ({len(arrays)})"
    return arrays, masks, final_filenames



def evaluate_correlation(dataset_config, uq_method, unc_measure):
    cfg = load_dataset_config(dataset_config)
    dataset_name = cfg['dataset_name']
    model_name = cfg['model_name']

    arrays, masks , filenames= load_unc_data(cfg, uq_method, unc_measure)

    # Print info
    print("____________________")
    print(f"Evaluating correlation matrix")
    print(f"Config: {model_name}")
    print(f"UQ method: {uq_method}")
    print(f"Uncertainty measure: {unc_measure}")
    print(f"Number of uncertainty maps: {len(arrays)}")
    print("____________________")

    # Create UncertaintyMap summary
    # TODO: Maybe batch bc not all 2000 arrays + masks should be loaded at once into the summary...
    # TODO: Solve empty values in correlation amtrix for high thresholds
    uq_maps = [UncertaintyMap(array=array, mask=mask, name=name) for array, mask, name in zip(arrays, masks, filenames)]
    summary = AggregationSummary(focus_strategy_list, name=f"{dataset_name}_summary", num_cpus=max_num_cpus())
    summary_df = summary.apply_methods(uq_maps, save_to_excel=False, do_plot=False, max_value=1.0)

    # Compute the correlation matrix
    corr_matrix = to_correlation_matrix(summary_df)
    out_name = f"correlation_matrix_{dataset_name}_{model_name}_{uq_method}_{unc_measure}"

    # Save to csv
    corr_matrix.to_csv(os.path.join("output", "tables", f"{out_name }.csv"))
    print(f"Correlation matrix {out_name}.csv saved to output folder.")

    # Save heatmap as png
    save_correlation_matrix_plot(corr_matrix, out_name, os.path.join("output", "figures"))
    print(f"Correlation heatmap {out_name}.png saved to output folder.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create accuracy-rejection curves for aggregators')
    parser.add_argument('--config', type=str, default='configs/ade20k_deeplab.yaml', help='Path to config file')
    parser.add_argument('--uq_method', type=str, default='dropout', help='UQ method: tta, softmax, ensemble, dropout')
    parser.add_argument('--unc_measure', type=str, default='pu', help='Uncertainty measure: pu, au or eu')
    args = parser.parse_args()
    
    evaluate_correlation(args.config, args.uq_method, args.unc_measure)

    

