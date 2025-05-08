import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp

from sklearn.metrics import roc_curve, auc
from functools import partial
from pathlib import Path
from PIL import Image
from matplotlib.patches import Patch
from tqdm import tqdm

from aggrigator.methods import AggregationMethods as am
from aggrigator.uncertainty_maps import UncertaintyMap
from aggrigator.optimized_gearys import fast_gearys_C
from data_utils import load_dataset, load_unc_maps, rescale_maps
from constants import (
    AUROC_STRATEGIES,
    NOISE_LEVELS,
    BARPLOTS_COLORS
    )

def setup_plot_style() -> None:
    """Sets up the plot style using tueplots and custom configurations."""
    plt.rcParams["text.latex.preamble"] += (
        r"\usepackage{amsmath} \usepackage{amsfonts} \usepackage{bm}"
    )

def preload_uncertainty_maps(uq_path, metadata_path, gt_list, task, model_noise, variation, data_noise):
    """Preload all uncertainty maps for a given noise level."""
    uq_methods = ['softmax', 'ensemble', 'dropout', 'tta']
    idx_task = 2 if task == 'instance' else 1
    gt_array = np.array(gt_list)[..., idx_task]
    
    # Dictionary to store loaded maps for each UQ method
    cached_maps = {}
    
    for uq_method in uq_methods:
        # Load zero-risk and noisy uncertainty maps
        uq_maps_zr, metadata_file_zr = load_unc_maps(uq_path, task, model_noise, variation, '0_00', 
                                                   uq_method, 'pu', False, metadata_path)
        uq_maps_r, metadata_file_r = load_unc_maps(uq_path, task, model_noise, variation, data_noise, 
                                                 uq_method, 'pu', False, metadata_path)
        
        # Normalize when needed
        uq_maps_zr = rescale_maps(uq_maps_zr, uq_method, task)
        uq_maps_r = rescale_maps(uq_maps_r, uq_method, task)
        
        # Concatenate maps
        uq_maps = np.concatenate((uq_maps_zr, uq_maps_r), axis=0)
        
        # Setup context masks
        context_gt = np.concatenate([gt_array, gt_array], axis=0)
        
        # Create UncertaintyMap objects
        uncertainty_maps = [UncertaintyMap(array=array, mask=gt, name=None) 
                           for (array, gt) in zip(uq_maps, context_gt)]
        
        # Define iD and OoD targets
        gt_labels_0 = np.zeros((len(uq_maps_zr)))
        gt_labels_1 = np.ones((len(uq_maps_r)))
        gt_labels = np.concatenate((gt_labels_0, gt_labels_1), axis=0)
        
        # Store in cache
        cached_maps[uq_method] = {
            'maps': uncertainty_maps,
            'gt_labels': gt_labels,
            'metadata': [metadata_file_zr, metadata_file_r]
        }
    
    return cached_maps

def compute_auroc_from_preloaded(cached_maps, uq_method, method, param, category):
    """Compute AUROC using preloaded uncertainty maps."""
    
    uncertainty_maps = cached_maps[uq_method]['maps']
    gt_labels = cached_maps[uq_method]['gt_labels']
    
    # Apply aggregation method
    uncertainty_values = np.array([method(map, param) for map in uncertainty_maps])
    
    # Handle threshold methods
    if category == 'Threshold':
        uncertainty_values = np.nan_to_num(uncertainty_values, nan=0)
        mask = (uncertainty_values == -1) | (uncertainty_values == 0)
        uncertainty_values[mask] = 0
    
    # Calculate AUROC
    fpr, tpr, _ = roc_curve(gt_labels, uncertainty_values)
    roc_auc = auc(fpr, tpr)
    
    if uq_method == 'softmax' and category == 'Threshold': 
        print(roc_auc)

    return roc_auc

def process_noise_level(uq_path, metadata_path, gt_list, task, model_noise, variation, noise_level, decomp):
    """Process all strategies for a single noise level."""
    print(f"Processing noise level: {noise_level}")
    
    # Preload all uncertainty maps for this noise level
    cached_maps = preload_uncertainty_maps(
        uq_path, metadata_path, gt_list, task, model_noise, variation, noise_level
    )
    
    auroc_data = []
    uq_methods = ['softmax', 'ensemble', 'dropout', 'tta']
    
    # Process each aggregation strategy
    for category, methods in AUROC_STRATEGIES.items():
        for aggr_name, (aggr_method, param) in tqdm(methods.items(), desc=f"Aggregator for noise {noise_level}"):
            try:
                print(f"----Processing aggregator function: {aggr_name}, in {category} category----")
                
                # Compute AUROC for each UQ method using preloaded maps
                auroc_values = np.zeros(len(uq_methods))
                for idx, uq_method in enumerate(uq_methods):
                    auroc_values[idx] = compute_auroc_from_preloaded(
                        cached_maps, uq_method, aggr_method, param, category
                    )
                
                # Store results
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
    df = df.sort_values('AUROC', ascending=False).reset_index(drop=True)
    print(df)
    
    path = os.getcwd()
    csv_file = f"{path}/output/tables/{task}_auroc_ood_detect_results.csv"

    # Check if the file exists to handle headers properly
    try:
        with open(csv_file, 'r') as f:
            file_empty = f.tell() == 0  # Check if file is empty
    except FileNotFoundError:
        file_empty = True  # If file doesn't exist, it's empty

    # Append to CSV (write header only if the file is empty)
    df.to_csv(csv_file, mode='a', index=False, header=file_empty)
    print(f"Data appended to {csv_file}")
    
    return df

def create_auroc_aggr_bar_plots(task, variation, uq_path, metadata_path, data_path, 
                                model_noise=0, decomp="pu"):
    """Create comparative bar plots of image-level AUROC values for different noise levels and UQ methods."""
    
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
    results = []
    for idx, noise_level in enumerate(NOISE_LEVELS):
        # Process noise level and get results
        df = process_noise_level(
            uq_path, metadata_path, gt_list, task, 
            model_noise, variation, noise_level, decomp
        )
        results.append(df)
    
    # Create plots for each noise level
    for idx, (noise_level, df) in enumerate(zip(NOISE_LEVELS, results)):
        ax = axes[idx]
        
        # Create a mapping from each method to its high-level category
        method_to_category = {method: category for category, methods in AUROC_STRATEGIES.items() for method in methods.keys()}

        bars = ax.bar(
            df['Aggregator'], 
            df['AUROC'],
            yerr=df['AUROC_std'],
            color=[BARPLOTS_COLORS[method_to_category[m]] for m in df['Aggregator']],
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
    legend_elements = [Patch(facecolor=v, label=k) for k, v in BARPLOTS_COLORS.items()]
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