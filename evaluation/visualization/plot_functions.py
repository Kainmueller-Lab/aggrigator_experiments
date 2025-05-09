import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from matplotlib.patches import Patch

# ---- Visualization Functions ----

def setup_plot_style() -> None:
    """
    Set up the plot style using custom configurations.
    """
    plt.rcParams["text.latex.preamble"] += (
        r"\usepackage{amsmath} \usepackage{amsfonts} \usepackage{bm}"
    )

def create_auroc_barplot(
    results: List[pd.DataFrame],
    noise_levels: List[str],
    barplot_colors: Dict[str, str],
    strategies_dict: Dict,
    task: str,
    variation: str,
    output_path: Path
    ) -> None:
    """
    Create comparative bar plots of image-level AUROC values.
    
    Parameters
    ----------
    results : List[pd.DataFrame]
        List of DataFrames with AUROC results for each noise level
    noise_levels : List[str]
        List of noise levels
    barplot_colors : Dict[str, str]
        Dictionary mapping categories to colors
    strategies_dict : Dict
        Dictionary of strategies by category
    task : str
        Task type ('instance' or 'semantic')
    variation : str
        Variation type
    output_path : Path
        Path to save the output figure
    """
    # Create figure with subplots in a single row
    fig, axes = plt.subplots(1, len(noise_levels), figsize=(20, 5))
    
    # Create a mapping from each method to its high-level category
    method_to_category = {
        method: category 
        for category, methods in strategies_dict.items() 
        for method in methods.keys()
    }
    
    # Create plots for each noise level
    for idx, (noise_level, df) in enumerate(zip(noise_levels, results)):
        ax = axes[idx]
        
        bars = ax.bar(
            df['Aggregator'], 
            df['AUROC'],
            yerr=df['AUROC_std'],
            color=[barplot_colors[method_to_category[m]] for m in df['Aggregator']],
            capsize=4,
            zorder=3,
        )
        
        # Add AUROC values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom'
            )
            
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
                fontsize=15,
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
    legend_elements = [
        Patch(facecolor=v, label=k) 
        for k, v in barplot_colors.items()
    ]
    fig.legend(
        handles=legend_elements, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 0.05),
        fancybox=True, 
        shadow=True, 
        ncol=3
    )
    
    # Add title
    plt.suptitle(
        f'OOD correctness measured by the AUROC w.r.t. model confidence correctness.\n'
        f'Task: {task}, Variation: {variation}', 
        fontsize=16
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    
    # Ensure output directory exists
    output_file = output_path.joinpath(f'output/figures/aggregators_auroc_{task}_{variation}_barplots.png')
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")