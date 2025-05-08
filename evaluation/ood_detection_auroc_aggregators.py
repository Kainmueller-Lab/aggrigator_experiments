import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import multiprocessing as mp

from sklearn.metrics import roc_curve, auc
from functools import partial
from pathlib import Path
from PIL import Image
from matplotlib.patches import Patch
from tqdm import tqdm

current_dir = Path.cwd()
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))
utils_dir = current_dir.parent / 'AggroUQ' / 'src'
sys.path.append(str(utils_dir))

from aggrigator.methods import AggregationMethods as am
from aggrigator.uncertainty_maps import UncertaintyMap
from aggrigator.optimized_gearys import fast_gearys_C
from evaluation.data_utils import load_dataset

COLOR = {
    'Baseline': "#A31212",
    'Context-aware' : "#BDB76B",
    'Threshold': sns.light_palette("blue", n_colors=6)[1],
    'Quantile': sns.light_palette("blue", n_colors=6)[2],
    'Patch': sns.light_palette("blue", n_colors=6)[3],
}

def setup_plot_style() -> None:
    """Sets up the plot style using tueplots and custom configurations."""
    plt.rcParams["text.latex.preamble"] += (
        r"\usepackage{amsmath} \usepackage{amsfonts} \usepackage{bm}"
    )
    
def preprocess_slice_spatial_measure(maps):
    """Ensures that hetmaps do not have negative values"""
    return np.clip(np.array(maps), 0, None)
    
def load_unc_maps_and_metadata(uq_path, metadata_path, task, model_noise, 
                                variation, data_noise, uq_method, decomp):
    """Load uncertainty maps and their corresponding metadata indices."""
    # Load uncertainty maps
    map_type = f"{task}_noise_{model_noise}_{variation}_{data_noise}_{uq_method}_{decomp}.npy"
    map_file = uq_path.joinpath(map_type)
    print(f"Loading uncertainty map: {map_type}")
    
    # uq_maps = preprocess_slice_spatial_measure(np.load(map_file))
    uq_maps = np.load(map_file)
    
    # Load metadata indices
    metadata_type = f"{task}_noise_{model_noise}_{variation}_{data_noise}_{uq_method}_{decomp}_sample_idx.npy"
    metadata_file_path = metadata_path.joinpath(metadata_type)
    indices = np.load(metadata_file_path)
        
    return uq_maps, indices

def rescale_maps(unc_map, rescale_fact, uq_method):
    if uq_method == 'softmax':
        return unc_map
    return unc_map / rescale_fact 

def compute_auroc(uq_path, metadata_path, gt_list, task, model_noise, uq_method, 
                     decomp, variation, data_noise, method, param, category):
    """Aggregate uncertainty values and compute AUROC correctness between scalars and ID/OOD samples over UQ methods."""
        
    # Load zero-risk and noisy uncertainty maps and metadata
    uq_maps_zr, metadata_file_zr = load_unc_maps_and_metadata(uq_path, metadata_path, task, model_noise, 
                                                              variation, '0_00', uq_method, decomp)
    uq_maps_r, metadata_file_r = load_unc_maps_and_metadata(uq_path, metadata_path, task, model_noise, 
                                                            variation, data_noise, uq_method, decomp)
    rescale_fact = np.log(3) if task == 'instance' else  np.log(6)
    uq_maps_zr = rescale_maps(uq_maps_zr, rescale_fact, uq_method)
    uq_maps_r = rescale_maps(uq_maps_r, rescale_fact, uq_method)
    uq_maps = np.concatenate((uq_maps_zr, uq_maps_r), axis=0)   
    
    # Define masks for context-aware aggregators 
    idx_task = 2 if task == 'instance' else 1
    gt_array = np.array(gt_list)[..., idx_task]  
    context_gt = np.concatenate([gt_array, gt_array], axis=0)
    
    # Apply aggregation strategies
    uq_maps = [UncertaintyMap(array=array, mask=gt, name=None) for (array, gt) in zip(uq_maps, context_gt)]
    metadata_file = [metadata_file_zr, metadata_file_r] #Tbd if necessary because the samples now appear correctly loaded for the same index 
    
    # Define iD and OoD targets for each sample
    gt_labels_0 = np.zeros((len(uq_maps_zr)))
    gt_labels_1 = np.ones((len(uq_maps_r)))
    gt_labels = np.concatenate((gt_labels_0, gt_labels_1), axis=0)
    
    uncertainty_values = np.array([method(map, param) for map in uq_maps])
    if category == 'Threshold':
        uncertainty_values = np.nan_to_num(uncertainty_values, nan=0)
        mask = (uncertainty_values == -1) | (uncertainty_values == 0)
        uncertainty_values[mask] = 0
    
    fpr, tpr, _ = roc_curve(gt_labels, uncertainty_values)
    roc_auc = auc(fpr, tpr)
    
    if uq_method == 'softmax' and category == 'Threshold': print(roc_auc)

    return roc_auc
    
def load_unc_and_compute_aggr_auroc(uq_path, metadata_path, gt_list, task, model_noise, variation, 
                                    data_noise, aggr_method, param, decomp, category):
    """Load uncertainty maps, aggregate them, define ID and OOD ground truth and calculate AUROC for each image."""
    
    uq_methods = ['softmax', 'ensemble', 'dropout', 'tta']
    methods_to_process = len(uq_methods)
    print(f"Processing {methods_to_process} uq methods")

    auroc_values = np.zeros(len(uq_methods),)
    for idx, unc_meth in enumerate(uq_methods):
        auroc_val = compute_auroc(
                uq_path, metadata_path, gt_list, task, model_noise, unc_meth, decomp, 
                variation, data_noise, aggr_method, param, category
        )
        auroc_values[idx] = auroc_val
    
    return auroc_values

def create_auroc_aggr_bar_plots(task, variation, uq_path, metadata_path, data_path,
                                model_noise=0, decomp="pu"):
    """Create comparative bar plots of image-level AUROC values for different noise levels and UQ methods."""
    
    # Constants
    noise_levels = ["0_25", "0_50", "0_75", "1_00"]
    strategies = {
        'Baseline': {
                'Mean': (am.mean, None), 
            },
        'Context-aware': {
                'Equally-w. class avg.' : (am.class_mean_w_equal_weights, None),
                'Imbalance-w. class avg.': (am.class_mean_weighted_by_occurrence, None),
                        },
        'Threshold':{
                'Threshold 0.3': (am.above_threshold_mean, 0.3),
                'Threshold 0.4': (am.above_threshold_mean, 0.4),
                'Threshold 0.5': (am.above_threshold_mean, 0.5),
            },
        'Quantile':{
                'Quantile 0.6': (am.above_quantile_mean, 0.6),
            'Quantile 0.75': (am.above_quantile_mean, 0.75),
            'Quantile 0.9': (am.above_quantile_mean, 0.9),
            },
        'Patch':{
                'Patch 10': (am.patch_aggregation, 10), 
                'Patch 20': (am.patch_aggregation, 20),
                'Patch 50': (am.patch_aggregation, 50),
            }
    }
    # Define paths
    output_path = Path(os.getcwd())
    
    # Load zero-risk labels 
    _, gt_list = load_dataset(
        data_path, 
        '0_00',
        is_ood='ood',
        num_workers=2,
        dataset_name='arctique'
    )
    
    # Create figure with subplots in a single row
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Process each noise level
    for idx, noise_level in enumerate(noise_levels):
        print(f"Processing noise level: {noise_level}")
        
        auroc_data = []
        for category, methods in strategies.items():
            for aggr_name, (aggr_method, param) in tqdm(methods.items(), desc=f"Aggregator for noise {noise_level}"):
                try:   
                    print(f"----Processing aggregator function: {aggr_name}, in {category} category----")
                    
                    auroc_values = load_unc_and_compute_aggr_auroc(
                        uq_path, metadata_path, gt_list, task, 
                        model_noise, variation, noise_level, 
                        aggr_method, param, decomp, category
                    )
                    
                    # Store mean AUROC value for this method and noise level
                    auroc_data.append({
                        'Aggregator': aggr_name,
                        'AUROC': np.mean(auroc_values),
                        'AUROC_std': np.std(auroc_values),
                        'Noise_Level': noise_level
                    })

                except Exception as e:
                    print(f"Error processing method {aggr_method} for noise level {noise_level}: {e}")
                    continue
        
        # Convert to DataFrame and sort by AUROC
        df = pd.DataFrame(auroc_data)
        # df['Aggregator'] = pd.Categorical(df['Aggregator'], categories=df['Aggregator'], ordered=True)
        df = df.sort_values('AUROC', ascending=False).reset_index(drop=True)
        print(df)
        
        path = os.getcwd()
        csv_file = f"{path}/output/{task}_auroc_ood_detect_results.csv"

        # Check if the file exists to handle headers properly
        try:
            with open(csv_file, 'r') as f:
                file_empty = f.tell() == 0  # Check if file is empty
        except FileNotFoundError:
            file_empty = True  # If file doesn't exist, it's empty

        # Append to CSV (write header only if the file is empty)
        df.to_csv(csv_file, mode='a', index=False, header=file_empty)
        print(f"Data appended to {csv_file}")
            
        # Create bar plot for this noise level
        ax = axes[idx]
        # Create a mapping from each method to its high-level category
        method_to_category = {method: category for category, methods in strategies.items() for method in methods.keys()}

        bars = ax.bar(
            df['Aggregator'], 
            df['AUROC'],
            yerr=df['AUROC_std'],
            # width=0.6,
            color=[COLOR[method_to_category[m]] for m in df['Aggregator']],
            capsize=4,
            zorder=3,
        )
        
        # Add AUROC values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
            
        # Add method label inside the bar, rotated horizontally    
        for bar, label in zip(bars, df['Aggregator']):
            y_offset = 0.005 * 2 * bar.get_height()  # Adjust offset as needed
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_offset,
                label,
                ha="center",
                va="bottom",
                rotation="vertical",
                fontsize=15,     # Font size
                zorder=4,
            )
        
        ax.set_title(f'Noise Level: {noise_level}')
        ax.set_ylabel('AUROC' + r" $\uparrow$", fontsize=12)
        ax.set_ylim(0, 1)  # AUROC is between 0 and 1
        ax.spines[['right', 'top']].set_visible(False)
        ax.tick_params(axis='y', which='major', labelsize=13)
        ax.set(xticklabels=[])
        ax.tick_params(bottom=False)   
    
    # Create legend
    legend_elements = [Patch(facecolor=v, label=k) for k, v in COLOR.items()]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05),
              fancybox=True, shadow=True, ncol=3)
    
    plt.suptitle(f'OOD correctness measured by the AUROC w.r.t. model confidence correctness.\nTask: {task}, Variation: {variation}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    
    # Save the plot
    output_file = output_path.joinpath(f'output/figures/aggregators_auroc_{task}_{variation}_barplots.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
def clear_csv_file(task):
    """Clears the content of the CSV file if it exists."""
    path = os.getcwd()
    csv_file = f"{path}/output/tables/{task}_auroc_ood_detect_results.csv"

    if os.path.exists(csv_file):  # Check if the file exists
        open(csv_file, 'w').close()  # Open in write mode to clear contents
        print(f"Cleared content of {csv_file}")
    else:
        print(f"{csv_file} does not exist yet.")
        
def parse_args():
    parser = argparse.ArgumentParser(description='Create ranked AUROC plots of unc. heatmaps at image level')
    parser.add_argument('--task', type=str, default='instance', help='Task type (e.g., instance or semantic)')
    parser.add_argument('--variation', type=str, default='nuclei_intensity', help='Variation type (e.g., nuclei_intensity or blood_cells)')
    parser.add_argument('--uq_path', type=str, default='/home/vanessa/Documents/data/uncertainty_arctique_v1-0-corrected_14/', help='Path to unc. evaluation results')
    parser.add_argument('--label_path', type=str, default='/home/vanessa/Desktop/synth_unc_models/data/v1-0-variations/variations/', help='Path to labels')
    parser.add_argument('--model_noise', type=int, default=0, help='Model noise level')
    parser.add_argument('--decomp', type=str, default='pu', help='Decomposition component (e.g. pu, au, eu)')
    
    return parser.parse_args()

if __name__ == "__main__":
    setup_plot_style()
    
    args = parse_args()
    
    #Clean Excel file for plot
    clear_csv_file(args.task)
    
    # Convert string paths to Path objects
    uq_path = Path(args.uq_path).joinpath("UQ_maps")
    data_path = Path(args.label_path).joinpath(args.variation) 
    metadata_path = Path(args.uq_path).joinpath("UQ_metadata")
    
    # Make sure output directory exists
    output_dir = Path(os.getcwd()).joinpath('output')
    output_dir.mkdir(exist_ok=True)
    
    create_auroc_aggr_bar_plots(
        task=args.task,
        variation=args.variation,
        uq_path=uq_path,
        metadata_path=metadata_path,
        data_path=data_path,
        model_noise=args.model_noise,
        decomp=args.decomp
    )