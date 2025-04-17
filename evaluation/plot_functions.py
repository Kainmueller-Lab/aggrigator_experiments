import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional 
from pathlib import Path
from constants import CLASS_NAMES_ARCTIQUE

# ---- Visualization Functions ----

def plot_acc_rej(
    portion_vals: np.ndarray, 
    acc_portion: np.ndarray, 
    cl_acc_portion: List[Dict[str, float]], 
    uq_method: str, 
    task: str, 
    variation: str, 
    output_path: Path, 
    aggregator_type: str, 
    method_names: Optional[List[str]] = None
    ) -> Path:
    """
    Create and save accuracy-rejection curve plots for aggregator methods.
    
    Args:
        portion_vals: rejection rate of accuracy based on tau most uncertain images 
        acc_portion: matrix (nxm) of n accuracy values for m aggregators 
        cl_acc_portion: list of accuracies by class, progressive with tau proportion 
        uq_method: UQ method (e.g., 'softmax', 'tta', 'ensemble')
        task: task name (e.g., 'semantic' or 'instance')
        variation: variation name (e.g., 'blood_cells')
        output_path: output dir.
        aggregator_type: class of aggregators used (e.g., 'class' or 'summary')
        method_names: names of subcategory of aggregators
    """
    
    num_methods = acc_portion.shape[1]  # Number of methods (columns)
    
    primary_colors = ['#1F77B4', '#6A0DAD', '#D62728', '#2CA02C', '#9467BD']
    lilac_shades = ['#D8A7D3', '#D77AB7', '#C7438B', '#9A4F89', '#701F56']
    
    color_list = primary_colors[:num_methods]
    if num_methods > len(color_list):
        color_list = primary_colors * (num_methods // len(primary_colors) + 1)
        color_list = color_list[:num_methods]
    
    # color_list = color_list + lilac_shades
    
    if method_names is None: # Default names
        method_names = [f"Method {i+1}" for i in range(num_methods)]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    
    # Plot 1: Aggregated sem. F1-rejection curves for each aggregator
    for i in range(num_methods):
        axes[0].plot(portion_vals[:-1], 
                     acc_portion[:-1, i], 
                     label=method_names[i], 
                     color=color_list[i], 
                     linestyle='-', 
                     linewidth=2)
    
    axes[0].set_xlabel(r"$\tau$" + " % Rejection " + r"$\uparrow$")
    axes[0].set_ylabel("Semantic Acc. (F1)")
    
    # Plot 2: Class-specific accuracy-rejection curves
    for key_index, key in enumerate(CLASS_NAMES_ARCTIQUE.keys()):
        # Extract the values for this key across all portions
        key_values = [cl_acc[key] for cl_acc in cl_acc_portion]  
        axes[1].plot(portion_vals[:-1], 
                     key_values[:-1], 
                     label=key, 
                     color=lilac_shades[key_index % len(lilac_shades)],
                     linestyle='--'
                     )

    axes[1].set_xlabel(r"$\tau$" + " % Rejection " + r"$\uparrow$")
    
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.005), ncol=5, fontsize=10)
    fig.tight_layout()

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.grid(True, which='both', axis='both', linestyle='--', color='gray', alpha=0.3) 
    
    # Save the plot
    output_file = output_path.joinpath(f'acc_rej_plot_{aggregator_type}_aggr_{uq_method}_{task}_{variation}_id.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")