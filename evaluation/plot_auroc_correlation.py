import numpy as np

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import seaborn as sns
import pandas as pd
import multiprocessing as mp

from sklearn.metrics import roc_curve, auc
from functools import partial
from pathlib import Path
from numpy import linalg as la
from PIL import Image
from matplotlib.patches import Patch
from tqdm import tqdm

NUCL_INT = {
    '1_00': '0_00',
    '0_75': '0_175',
    '0_50' : '0_35', 
    '0_25': '0_525',
    '0_00': '0_70'
}

def setup_plot_style() -> None:
    """Sets up the plot style using tueplots and custom configurations."""
    plt.rcParams["text.latex.preamble"] += (
        r"\usepackage{amsmath} \usepackage{amsfonts} \usepackage{bm}"
    )

def manual_auroc_comput(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes the Area Under the Receiver Operating Characteristic curve (AUROC) using NumPy.
    Adaptation of the function contained in  https://github.com/bmucsanyi/untangle/blob/main/untangle/utils/metric.py 

    Args:
        y_true: True binary labels.
        y_score: Target scores.

    Returns:
        The AUROC score.
    """
    # Sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # Compute the AUC
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.concatenate([
        distinct_value_indices,
        np.array([y_true.size - 1]),
    ])

    true_positives = np.cumsum(y_true)[threshold_idxs]
    false_positives = 1 + threshold_idxs - true_positives

    true_positives = np.concatenate([
        np.array([0]),
        true_positives,
    ])
    false_positives = np.concatenate([
        np.array([0]),
        false_positives,
    ])

    if false_positives[-1] <= 0 or true_positives[-1] <= 0:
        return np.nan

    false_positive_rate = false_positives / false_positives[-1]
    true_positive_rate = true_positives / true_positives[-1]

    return np.trapz(true_positive_rate, false_positive_rate)

def process_single_image_with_auroc(idx, metadata_file, uq_maps, images_path, variation, data_noise):
    """Process a single image and and calculate AUROC between its predictive unc. values and unc. ground truth pixels."""
    
    # Load OOD and ID images    
    data_noise_img = data_noise if variation == 'blood_cells' else NUCL_INT[data_noise]
    zero_risk_data = "0_00" if variation == 'blood_cells' else NUCL_INT["0_00"]
    
    img_file_ood = Path(images_path).joinpath(f"{data_noise_img}/images/img_{metadata_file[idx]}.png")
    img_file_id = Path(images_path).joinpath(f"{zero_risk_data}/images/img_{metadata_file[idx]}.png")
    
    img_ood = np.array(Image.open(img_file_ood)).astype(np.float32)[:,:,:3]
    img_id = np.array(Image.open(img_file_id)).astype(np.float32)[:,:,:3]
    
    # Calculate ground truth unc. pixels
    gt_ood = img_ood - img_id
    ood_norm = la.norm(gt_ood, 2, 2)
    threshold = (np.max(ood_norm) - np.min(ood_norm))/2 if variation == 'blood_cells' else 0.1
    gt_ood = (ood_norm > threshold).astype(int)
    
    # Get uncertainty values and corresponding ground truth labels
    uncertainty_values = uq_maps[idx].flatten()
    gt_labels = gt_ood.flatten()

    fpr, tpr, _ = roc_curve(gt_labels, uncertainty_values)
    roc_auc = auc(fpr, tpr)

    return roc_auc

def load_and_compute_auroc(uq_path, metadata_path, images_path, task, model_noise, 
                          variation, data_noise, uq_method, decomp, num_samples=30, n_processes=None):
    """Load uncertainty maps, process corresponding ground truth and calculate AUROC for each image."""
    # Load uncertainty maps
    map_type = f"{task}_noise_{model_noise}_{variation}_{data_noise}_{uq_method}_{decomp}.npy"
    map_file = uq_path.joinpath(map_type)
    print(f"Loading UQ map from: {map_file}")  # Debugging line
    uq_maps = np.load(map_file)
    
    # Load metadata
    metadata_type = f"{task}_noise_{model_noise}_{variation}_{data_noise}_{uq_method}_{decomp}_sample_idx.npy"
    metadata_file_path = metadata_path.joinpath(metadata_type)
    metadata_file = np.load(metadata_file_path)
    
    # Determine how many samples to process
    total_available = len(metadata_file)
    samples_to_process = min(num_samples, total_available)
    print(f"Processing {samples_to_process} samples out of {total_available} available")
    
    # Determine number of processes to use
    if n_processes is None:
        n_processes = mp.cpu_count()
    n_processes = min(n_processes, samples_to_process)  # Don't use more processes than samples
    
    # Create a partial function with fixed arguments
    process_func = partial(
        process_single_image_with_auroc,
        metadata_file=metadata_file,
        uq_maps=uq_maps,
        images_path=images_path,
        variation=variation,
        data_noise=data_noise
    )
    
    # Process images in parallel and calculate AUROC
    print(f"Calculating AUROC using {n_processes} processes")
    with mp.Pool(processes=n_processes) as pool:
        auroc_values = list(tqdm(
            pool.imap(process_func, range(samples_to_process)),
            total=samples_to_process,
            desc="Calculating AUROC"
        ))
    
    return auroc_values

def create_auroc_bar_plots(task, variation, data_path, uq_path, metadata_path, 
                           model_noise=0, decomp="pu", num_samples=30, n_processes=None):
    """Create comparative bar plots of AUROC values for different noise levels and UQ methods."""
    
    # Constants
    noise_levels = ["0_25", "0_50", "0_75", "1_00"]
    uq_methods = ["tta", "softmax", "ensemble"]
    
    # Define custom colors for each method
    method_colors = {
        "softmax": "lightgrey",
        "ensemble": "lightgreen", 
        "tta": "lightblue"
    }
    
    # Define paths
    images_path = Path(data_path).parent.joinpath(f'v1-0-variations/variations/{variation}/')
    output_path = Path(os.getcwd())
    
    # Create figure with subplots in a single row
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Process each noise level
    for idx, noise_level in enumerate(noise_levels):
        print(f"Processing noise level: {noise_level}")
        
        # Prepare data for plotting
        auroc_data = []
        for method in tqdm(uq_methods, desc=f"Methods for noise {noise_level}"):
            auroc_values = load_and_compute_auroc(
                uq_path, metadata_path, images_path,
                task, model_noise, variation, noise_level, method, decomp,
                num_samples, n_processes
            )
            
            # Store mean AUROC value for this method and noise level
            auroc_data.append({
                'Method': method,
                'AUROC': np.mean(auroc_values),
                'AUROC_std': np.std(auroc_values),
                'Noise_Level': noise_level
            })
        
        # Convert to DataFrame and sort by AUROC
        df = pd.DataFrame(auroc_data)
        df = df.sort_values('AUROC', ascending=False)
        
        # Create bar plot for this noise level
        ax = axes[idx]
        bars = ax.bar(
            df['Method'], 
            df['AUROC'],
            yerr=df['AUROC_std'],
            # width=0.6,
            capsize=4,
            color=[method_colors[method] for method in df['Method']],
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
        for bar, label in zip(bars, df['Method']):
            if label == 'softmax': label = label + " (baseline)"
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
        # ax.set_xlabel('UQ Method', fontsize=12)
        ax.set_ylabel('AUROC' + r" $\uparrow$", fontsize=12)
        ax.set_ylim(0, 1)  # AUROC is between 0 and 1
        ax.spines[['right', 'top']].set_visible(False)
        ax.tick_params(axis='y', which='major', labelsize=13)
        ax.set(xticklabels=[])
        ax.tick_params(bottom=False)        
    
    # Create legend
    legend_elements = [
        Patch(facecolor='lightgrey', label='Deterministic'),
        Patch(facecolor='lightgreen', label='Distributional'),
        Patch(facecolor='lightblue', label='Heuristic')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05),
              fancybox=True, shadow=True, ncol=3)
    
    plt.suptitle(f'AUROC Performance Comparison for Different UQ Methods\nTask: {task}, Variation: {variation}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.90])  # 
    
    # Save the plot
    output_file = output_path.joinpath(f'output/auroc_{task}_{variation}_barplot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
def parse_args():
    parser = argparse.ArgumentParser(description='Create ranked correlation plots of unc. heatmaps')
    parser.add_argument('--task', type=str, default='instance', help='Task type (e.g., instance or semantic)')
    parser.add_argument('--variation', type=str, default='nuclei_intensity', help='Variation type (e.g., nuclei_intensity or blood_cells)')
    parser.add_argument('--uq_path', type=str, default='/home/vanessa/Documents/data/uncertainty_arctique_v1-0-corrected_14/', help='Path to unc. evaluation results')
    parser.add_argument('--data_path', type=str, default='/home/vanessa/Desktop/synth_unc_models/data/v1-0-variations', help='Path to ID and OOD images')
    parser.add_argument('--model_noise', type=int, default=0, help='Model noise level')
    parser.add_argument('--decomp', type=str, default='pu', help='Decomposition component (e.g. pu, au, eu)')
    parser.add_argument('--num_samples', type=int, default=30, help='Number of samples to process')
    parser.add_argument('--n_processes', type=int, default=None, help='Number of processes to use (default: CPU count)')
    
    return parser.parse_args()

if __name__ == "__main__":
    setup_plot_style()
    
    args = parse_args()
    
    # Convert string paths to Path objects
    data_path = Path(args.data_path)
    uq_path = Path(args.uq_path).joinpath("UQ_maps")
    metadata_path = Path(args.uq_path).joinpath("UQ_metadata")
    
    # Set up multiprocessing start method for compatibility across platforms
    mp.set_start_method('spawn', force=True)
    
    create_auroc_bar_plots(
        task=args.task,
        variation=args.variation,
        data_path = data_path, 
        uq_path=uq_path,
        metadata_path=metadata_path,
        model_noise=args.model_noise,
        decomp=args.decomp,
        num_samples=args.num_samples,
        n_processes=args.n_processes
    )
    

