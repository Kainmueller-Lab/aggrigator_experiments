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
def process_single_image(idx, metadata_file, uq_maps, images_path, variation, data_noise):
    """Process a single image and return its predictive unc. values and unc. ground truth pixels."""
    
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
    threshold = (np.max(ood_norm) - np.min(ood_norm))/2
    gt_ood = (ood_norm > threshold).astype(int)
    
    # Get uncertainty values and corresponding ground truth labels
    uncertainty_values = uq_maps[idx].flatten()
    gt_labels = gt_ood.flatten()

    return uncertainty_values, gt_labels

def load_and_process_data(uq_path, metadata_path, images_path, task, model_noise, 
                          variation, data_noise, uq_method, decomp, num_samples=30, n_processes=None):
    """Load and process uncertainty maps and corresponding ground truth."""
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
        process_single_image,
        metadata_file=metadata_file,
        uq_maps=uq_maps,
        images_path=images_path,
        variation=variation,
        data_noise=data_noise
    )
    
    # Process images in parallel
    print(f"Processing images using {n_processes} processes")
    with mp.Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_func, range(samples_to_process)),
            total=samples_to_process,
            desc="Processing images"
        ))
        
    # Combine results
    uncertainty_values = []
    gt_labels = []
    for uv, gt in results:
        uncertainty_values.extend(uv)
        gt_labels.extend(gt)
    
    return np.array(uncertainty_values), np.array(gt_labels)

def create_comparative_plots(task, variation, data_path, uq_path, metadata_path, 
                             model_noise=0, decomp="pu", num_samples=30, n_processes=None):
    """Create comparative plots for different image noise levels and UQ methods.
    Args:
        task: Task type (default: 'instance')
        variation: Variation type (default: 'nuclei_intensity')
        data_path: Path to ID and OOD images
        uq_path: Path to UQ maps
        metadata_path: Path to metadata
        output_path: Path to save output plots (default: current directory)
        model_noise: Model noise level (default: 0)
        decomp: Decomposition component (default: 'pu')
        num_samples: Number of images to process (default: 30)
        n_processes: Number of processes to use (default: CPU count)
    """
    
    # Constants
    noise_levels = ["0_25", "0_50", "0_75", "1_00"]
    uq_methods = ["tta", "softmax", "ensemble"]
    
    # Define paths
    images_path = Path(data_path).parent.joinpath(f'v1-0-variations/variations/{variation}/')
    output_path = Path(os.getcwd())
     
    # Create figure with subplots in a single row
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.3))
    
    # Process each noise level
    df_plot = []
    for idx, noise_level in enumerate(noise_levels):
        print(f"Processing noise level: {noise_level}")
        
        # Prepare data for plotting
        plot_data = []
        for method in tqdm(uq_methods):
            values, labels = load_and_process_data(
                uq_path, metadata_path, images_path,
                task, model_noise, variation, noise_level, method, decomp,
                num_samples, n_processes
            )
            
            # Collect data for both ground truth cases
            plot_data.extend([
                {'Method': method, 'GT': 'ID pixels', 'Uncertainty': val}
                for val in values[labels == 0]
            ])
            plot_data.extend([
                {'Method': method, 'GT': 'OOD pixels', 'Uncertainty': val}
                for val in values[labels == 1]
            ])
        
        # Convert list of dictionaries to DataFrame
        df_current = pd.DataFrame(plot_data)
        
        # Calculate mean uncertainties for each method and GT category
        means = df_current.groupby(['GT', 'Method'])['Uncertainty'].mean().reset_index()
        
        # Sort methods by mean uncertainty for each GT category
        order_id = means[means['GT'] == 'ID pixels'].sort_values('Uncertainty', ascending=False)['Method'].tolist()
        order_ood = means[means['GT'] == 'OOD pixels'].sort_values('Uncertainty', ascending=False)['Method'].tolist()
        
        # Create a custom order combining both
        custom_order = list(dict.fromkeys(order_id + order_ood))
        
        # Reorder the DataFrame
        df_current['Method'] = pd.Categorical(df_current['Method'], categories=custom_order, ordered=True)
        df_current = df_current.sort_values('Method')
        
        # Store for plotting
        df_plot.append(df_current)
        
        flierprops = dict(marker='o', 
                          markerfacecolor='white', 
                          markeredgecolor='gray', 
                          markersize=6, 
                          linestyle='--', 
                          linewidth=.5)
        
        # Create grouped boxplot
        sns.boxplot(
            data=df_plot[idx],
            x='GT',       # Group by Ground Truth
            y='Uncertainty',
            hue='Method', # Different methods side by side within each GT category
            ax=axes[idx],
            showfliers=False, 
            # flierprops=flierprops,
            boxprops=dict(alpha=.6),
            palette="Set3"
            )
        
        axes[idx].set_title(f'Noise Level: {noise_level}')
        axes[idx].set_xlabel('Uncertainty Ground Truth', fontsize=12)
        axes[idx].set_ylabel('pu', fontsize=12)
        axes[idx].spines[['right', 'top']].set_visible(False)
        axes[idx].tick_params(axis='both', which='major', labelsize=13)
        axes[idx].set_yticks([0,1]); 
        
        axes[idx].legend(title='UQ Method', loc='upper right')
    
    plt.subplots_adjust(top=0.85)
    plt.tight_layout()
    output_file = output_path.joinpath(f'output/heatmaps_{task}_{variation}_ood_gt_histograms.png')
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
    args = parse_args()
    
    # Convert string paths to Path objects
    data_path = Path(args.data_path)
    uq_path = Path(args.uq_path).joinpath("UQ_maps")
    metadata_path = Path(args.uq_path).joinpath("UQ_metadata")
    
    # Set up multiprocessing start method for compatibility across platforms
    mp.set_start_method('spawn', force=True)
    
    create_comparative_plots(
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
    
