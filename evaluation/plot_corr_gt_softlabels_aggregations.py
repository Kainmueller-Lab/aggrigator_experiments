import numpy as np
import os
import sys
import argparse
import glob
import random
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

random.seed(42)

COLOR = {
    'Baseline': "#A31212",
    'Spatial' : "#BDB76B",
    'Threshold': sns.light_palette("blue", n_colors=6)[1],
    'Quantile': sns.light_palette("blue", n_colors=6)[2],
    'Patch': sns.light_palette("blue", n_colors=6)[3],
}

def setup_plot_style() -> None:
    """Sets up the plot style using tueplots and custom configurations."""
    plt.rcParams["text.latex.preamble"] += (
        r"\usepackage{amsmath} \usepackage{amsfonts} \usepackage{bm}"
    )
    
def entropy(probs: np.ndarray, axis: int = -1, eps: float = 1e-5) -> np.ndarray:
    """Computes the entropy of a probability distribution using NumPy."""
    log_probs = np.log(probs + eps)
    min_real = np.finfo(log_probs.dtype).min
    log_probs = np.clip(log_probs, a_min=min_real, a_max=None)
    return - (log_probs * probs).sum(axis=axis)
    
def preprocess_slice_spatial_measure(maps):
    """Ensures that hetmaps do not have negative values"""
    return np.clip(np.array(maps), 0, None)
    
def load_unc_maps_and_metadata(uq_path, metadata_path, task, model_noise, 
                                variation, noise_level, uq_method, decomp):
    """Load uncertainty maps and their corresponding metadata indices."""
    # Load uncertainty maps
    map_type = f"{task}_noise_{model_noise}_{variation}_{noise_level}_{uq_method}_{decomp}.npy"
    map_file = uq_path.joinpath(map_type)
    print(f"Loading uncertainty map: {map_type}")
    
    uq_maps = preprocess_slice_spatial_measure(np.load(map_file))
        
    # Load metadata indices
    metadata_type = f"{task}_noise_{model_noise}_{variation}_{noise_level}_{uq_method}_{decomp}_sample_idx.npy"
    metadata_file_path = metadata_path.joinpath(metadata_type)
    indices = np.load(metadata_file_path)
        
    return uq_maps, indices

def process_aggr_unc(uq_path, metadata_path, label_path, task, model_noise, uq_method, decomp, 
                     variation, noise_level, method, param):
    """Aggregate uncertainty values and compute AUROC correctness between scalars and ID/OOD samples over UQ methods."""
    
    # Load mask noisy uncertainty maps amnd metadata
    uq_maps, metadata_file = load_unc_maps_and_metadata(uq_path, metadata_path, task, model_noise, 
                                                            variation, noise_level, uq_method, decomp)
    uq_maps = [UncertaintyMap(array=array, mask=None, name=None) for array in uq_maps]
    
    # Define GT soft-label uncertainty
    zero_au_label_path = label_path + f'{task}_noise_0/*.png'
    au_label_path = label_path + f'{task}_noise_{model_noise}/*.png'

    masks_zero_au_files = glob.glob(zero_au_label_path)
    mask_noise_au_files = glob.glob(au_label_path)
    
    # Select 50 random samples from masks_zero_au
    random_samples = random.sample(masks_zero_au_files, 50)
    # Extract the filenames (without directory)
    selected_filenames = {os.path.basename(f) for f in random_samples}
    # Find corresponding files in mask_noise_au
    matching_masks = [f for f in mask_noise_au_files if os.path.basename(f) in selected_filenames]
    matching_masks_zero = [f for f in masks_zero_au_files if os.path.basename(f) in selected_filenames]
    
    mask_zero_au = np.array([np.array(Image.open(mask), dtype = int) for mask in matching_masks_zero])
    mask_noise_au = np.array([np.array(Image.open(mask), dtype = int) for mask in matching_masks])
    
    labels_combined = [np.concatenate([mz.flatten(), ms.flatten()]) for mz, ms in zip(mask_zero_au, mask_noise_au)]  # Flatten and merge labels
    
    entr_val = np.zeros((len(labels_combined)))
    for idx, label in enumerate(labels_combined):
        unique_labels, counts = np.unique(label, return_counts=True)
        probs = counts / counts.sum(-1)
        entr_val[idx] = entropy(probs)
        
    uncertainty_values = np.array([method(map, param) for map in uq_maps])    
    corr_val= np.corrcoef(np.stack([uncertainty_values, uncertainty_values]))[0, 1]

    return corr_val

def load_unc_and_compute_aggr_corr(uq_path, metadata_path, label_path, task, model_noise, variation, 
                                    noise_level, aggr_method, param, decomp):
    """Load uncertainty maps, aggregate them, define ID and OOD ground truth and calculate AUROC for each image."""
    
    uq_methods = ['softmax', 'tta', 'ensemble']
    methods_to_process = len(uq_methods)
    print(f"Processing {methods_to_process} uq methods")
    
    corr_values = np.zeros(len(uq_methods),)
    for idx, unc_meth in enumerate(uq_methods):
        corr_val = process_aggr_unc(
                uq_path, metadata_path, label_path, task, model_noise, 
                unc_meth, decomp, variation, noise_level, aggr_method, param
        )
        corr_values[idx] = corr_val
    
    return corr_values

def create_auroc_aggr_bar_plots(task, variation, uq_path, metadata_path, 
                                label_path, noise_level='0_00', decomp="pu"):
    """Create comparative bar plots of image-level correlation values for different masks' noise levels and UQ methods."""
        
    # Constants
    model_noise = [20, 40, 60, 80, 100]
    strategies = {
        'Baseline': {
                'Mean': (am.mean, None), 
                'Sum': (am.sum, None),
                'Max': (am.max, None),
            },
        'Spatial': {
                'Gearys': (fast_gearys_C, None),
                        },
        'Threshold':{
                'Threshold 0.3': (am.above_threshold_mean, 0.3),
                'Threshold 0.4': (am.above_threshold_mean, 0.4),
                'Threshold 0.5': (am.above_threshold_mean, 0.5),
            },
        'Quantile':{
                'Quantile 0.6': (am.above_quantile_mean, 0.6),
            'Quantile 0.75': (am.above_quantile_mean, 0.8),
            'Quantile 0.9': (am.above_quantile_mean, 0.6),
            },
        'Patch':{
                'Patch 10': (am.patch_aggregation, 10), 
                'Patch 20': (am.patch_aggregation, 20),
                'Patch 50': (am.patch_aggregation, 20),
            }
    }
    
    # Define paths
    output_path = Path(os.getcwd())
    
    # Create figure with subplots in a single row
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    
    # Process each noise level
    for idx, m_noise in enumerate(model_noise):
        print(f"Processing mask noise level: {m_noise}")
        
        corr_data = []
        for category, methods in strategies.items():
            for aggr_name, (aggr_method, param) in tqdm(methods.items(), desc=f"Aggregator for mask noise {m_noise}%"):
                try:   
                    print(f"----Processing aggregator function: {aggr_name}, in {category} category----")

                    corr_values = load_unc_and_compute_aggr_corr(
                        uq_path, metadata_path, label_path, task, m_noise, 
                        variation, noise_level, aggr_method, param, decomp
                    )
                    
                    # Store mean corr. value for this method and noise level
                    corr_data.append({
                        'Aggregator': aggr_name,
                        'Corr': np.mean(corr_values),
                        'Corr_std': np.std(corr_values),
                        'Mask_Noise_Level': m_noise
                    })

                except Exception as e:
                    print(f"Error processing method {aggr_method} for mask noise level {m_noise}: {e}")
                    continue
        
        # Convert to DataFrame and sort by AUROC
        df = pd.DataFrame(corr_data)
        # df['Aggregator'] = pd.Categorical(df['Aggregator'], categories=df['Aggregator'], ordered=True)
        df = df.sort_values('Corr', ascending=False).reset_index(drop=True)
        print(df)
            
        # Create bar plot for this noise level
        ax = axes[idx]
        # Create a mapping from each method to its high-level category
        method_to_category = {method: category for category, methods in strategies.items() for method in methods.keys()}

        bars = ax.bar(
            df['Aggregator'], 
            df['Corr'],
            yerr=df['Corr_std'],
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
        
        ax.set_title(f'Mask Noise Level: {m_noise}')
        ax.set_ylabel('Rank Corr.' + r" $\uparrow$", fontsize=12)
        ax.set_ylim(0, 1)  # AUROC is between 0 and 1
        ax.spines[['right', 'top']].set_visible(False)
        ax.tick_params(axis='y', which='major', labelsize=13)
        ax.set(xticklabels=[])
        ax.tick_params(bottom=False)   
    
    # Create legend
    legend_elements = [Patch(facecolor=v, label=k) for k, v in COLOR.items()]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05),
              fancybox=True, shadow=True, ncol=3)
    
    plt.suptitle(f'Rank correlation of aggregators measured by correlation w.r.t. GT aleatoric uncertainty.\nTask: {task}, Variation: {variation}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    
    # Save the plot
    output_file = output_path.joinpath(f'output/aggregators_rank_corr_{task}_{variation}_barplot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

def parse_args():
    parser = argparse.ArgumentParser(description='Create ranked correlation plots of unc. heatmaps at image level')
    parser.add_argument('--task', type=str, default='instance', help='Task type (e.g., instance or semantic)')
    parser.add_argument('--variation', type=str, default='nuclei_intensity', help='Variation type (e.g., nuclei_intensity or blood_cells)')
    parser.add_argument('--uq_path', type=str, default='/home/vanessa/Documents/data/uncertainty_arctique_v1-0-corrected_14/', help='Path to unc. evaluation results')
    parser.add_argument('--label_path', type=str, default='/home/vanessa/Desktop/synth_unc_models/data/v1-0-corrected_NOISY/train/masks/', help='Path to labels')
    parser.add_argument('--noise_level', type=str, default='0_00', help='Image noise level')
    parser.add_argument('--decomp', type=str, default='pu', help='Decomposition component (e.g. pu, au, eu)')
    
    return parser.parse_args()

if __name__ == "__main__":
    setup_plot_style()
    
    args = parse_args()
    
    # Convert string paths to Path objects
    uq_path = Path(args.uq_path).joinpath("UQ_maps")
    metadata_path = Path(args.uq_path).joinpath("UQ_metadata")
    
    # Make sure output directory exists
    output_dir = Path(os.getcwd()).joinpath('output')
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading labels in {args.label_path}")
    
    create_auroc_aggr_bar_plots(
        task=args.task,
        variation=args.variation,
        uq_path=uq_path,
        metadata_path=metadata_path,
        label_path=args.label_path, 
        noise_level=args.noise_level,
        decomp=args.decomp
    )
    