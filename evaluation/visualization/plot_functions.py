import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from matplotlib.patches import Patch

from evaluation.data_utils import AnalysisResults
from evaluation.constants import COLORS

# ---- Visualization Functions ----

def setup_plot_style_auroc() -> None:
    """
    Set up the AUROC barplots style using custom configurations.
    """
    plt.rcParams["text.latex.preamble"] += (
        r"\usepackage{amsmath} \usepackage{amsfonts} \usepackage{bm}"
    )

def setup_plot_style_aurc() -> None:
    """
    Set up the AURC plot style using custom configurations.
    """
    plt.rcParams["text.latex.preamble"] += (
        r"\usepackage{amsmath} \usepackage{amsfonts} \usepackage{bm}"
    )
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

def create_single_auroc_barplot(
    results: pd.DataFrame,
    barplot_colors: Dict[str, str],
    strategies_dict: Dict,
    task: str,
    variation: str,
    dataset_name : str,
    decomp: str,
    output_path: Path,
) -> None:
    """
    Create a single bar plot of image-level AUROC values.
    
    Parameters
    ----------
    results : pd.DataFrame
        DataFrame with AUROC results
    barplot_colors : Dict[str, str]
        Dictionary mapping categories to colors
    strategies_dict : Dict
        Dictionary of strategies by category
    task : str
        Task type ('instance' or 'semantic')
    variation : str
        Variation type
    dataset_name : str
        Dataset analyzed (e.g. 'lizard')
    decomp : str
        Uncertainty component tested ('pu', 'eu' or 'au')
    output_path : Path
        Path to save the output figure
    """
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    # Create a mapping from each method to its high-level category
    method_to_category = {
        method: category
        for category, methods in strategies_dict.items()
        for method in methods.keys()
    }
    
    # Create bar plot
    bars = ax.bar(
        results['Aggregator'],
        results['AUROC'],
        yerr=results['AUROC_std'],
        color=[barplot_colors[method_to_category[m]] for m in results['Aggregator']],
        capsize=4,
        zorder=3,
    )
    
    # Add AUROC values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f'{height:.3f}',
            xy=(bar.get_x() + bar.get_width()/2, height),
            xytext=(0, 3), # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom'
        )
    
    # Add method label inside the bar, rotated vertically
    for bar, label in zip(bars, results['Aggregator']):
        y_offset = 0.005 * 2 * bar.get_height() # Adjust offset as needed
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
    
    # Set labels and formatting
    ax.set_ylabel('AUROC' + r" $\uparrow$", fontsize=12)
    ax.set_ylim(0, 1) # AUROC is between 0 and 1
    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(axis='y', which='major', labelsize=13)
    ax.set(xticklabels=[])
    ax.tick_params(bottom=False)
    
    # Create legend
    legend_elements = [
        Patch(facecolor=v, label=k)
        for k, v in barplot_colors.items()
    ]
    ax.legend(
        handles=legend_elements, 
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.025),
        fancybox=True, 
        shadow=True, 
        ncol=3
    )
    
    # Add title
    title_text = (
        f'OOD correctness measured by the AUROC w.r.t. model confidence correctness.\n'
        f'Task: {task}, Variation: {variation}'
    )
    
    plt.title(title_text, fontsize=16, pad=20)
    plt.tight_layout()
    
    # Ensure output directory exists
    output_file = output_path.joinpath(f'figures/ood_auroc_{task}_{dataset_name}_{variation}_{decomp}_barplot.png')
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.close()  # Close figure to free memory
    
def create_auroc_barplot(
    results: List[pd.DataFrame],
    noise_levels: List[str],
    barplot_colors: Dict[str, str],
    strategies_dict: Dict,
    task: str,
    variation: str,
    dataset_name : str,
    decomp: str,
    output_path: Path,
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
    output_file = output_path.joinpath(f'figures/ood_auroc_{task}_{dataset_name}_{variation}_{decomp}_barplot.png')
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

def create_selective_risks_coverage_plot(
        method_names: List[str],
        aurc_res: AnalysisResults,
        output_path: Path, 
        args: argparse.Namespace
    ) -> None:
    """
    Create and save AURC plot.
    
    Args:
        method_names: List of method names
        augrc_res: Analysis results containing AURC data
        output_path: Path to save output
        args: Command line arguments
    """
    # # Plot mean results
    x = aurc_res.coverages.flatten() # Flatten to 1D for plotting
    y = aurc_res.mean_selective_risks # Shape: [coverage points, num_strategies]
    y_std = aurc_res.std_selective_risks # Shape: same as y
    
    # Prepare data dictionary for CSV export
    data_dict = {"Coverage": x[::-1]} # Reverse to match plotting order
    
    # Define method categories for styling
    method_categories = ["Threshold", "Patch", "Quantile"]
    first_occurrence = {cat: True for cat in method_categories}
    
    # Plot each method
    plt.figure(figsize=(8, 6))
    for j, method_name in enumerate(method_names):
        data_dict[f"{method_name} (Mean Risk)"] = y[:, j][::-1]
        data_dict[f"{method_name} (Std Dev)"] = y_std[:, j][::-1]
        
        color = COLORS[j % len(COLORS)]
        linestyle = '-'  # Default solid line
        alpha = 1.0  # Default opacity
        linewidth = 2  # Default line width
        alpha_fill_in = 0.2 #default fill-in transparency
        
        # Check if the method belongs to a category
        for cat in method_categories:
            if method_name.startswith(cat):
                if first_occurrence[cat]:
                    first_occurrence[cat] = False  # Mark first as used
                else:
                    linestyle = '--'  # Dashed line for subsequent ones
                    linewidth = 1 # Make it thinner
                    alpha = 0.5  # Make it more transparent
                    alpha_fill_in = 0.1
                break  # Exit loop once category is found
        
        if method_name.startswith("Mean"):
            color = 'gray'
            linewidth = 2
        
        plt.plot(x[::-1], y[:, j][::-1], 
                 label=f"{method_names[j]} (AURC: {aurc_res.mean_aurc[j]:.4f})",
                 linewidth=linewidth, color=color, linestyle=linestyle, alpha=alpha)
        
        # Add shaded area (mean ± std)
        plt.fill_between(x[::-1], 
                        (y[:, j] - y_std[:, j])[::-1],  # Lower bound
                        (y[:, j] + y_std[:, j])[::-1],  # Upper bound
                        color=color, alpha=alpha_fill_in)  # Transparency
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data_dict)
    
    # Define output file paths
    ood = 'ood' if args.image_noise != '0_00' else 'id'
    csv_file = output_path.joinpath(
        f'tables/aurc_data_{args.aggregator_type}_aggr_multi_uq_methods_{args.task}_{args.variation}_{ood}.csv'
    )
    
    # Check if file exists to handle headers
    file_empty = not csv_file.exists() or csv_file.stat().st_size == 0
    df.to_csv(csv_file, mode='a', index=False, header=file_empty)
    print(f"Data saved to: {csv_file}")
    
    # Finalize plot
    plt.xlabel("Coverage")
    plt.ylabel("Selective Risks")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=8)
    plt.grid(False)
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save plot
    output_file = output_path.joinpath(
        f'figures/{ood}_aurc_{args.task}_{args.dataset_name}_{args.variation}_{args.decomp}.png'
    )
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")