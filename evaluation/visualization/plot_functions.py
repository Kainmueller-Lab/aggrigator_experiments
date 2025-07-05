import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple
from matplotlib.patches import Patch

from evaluation.data_utils import AnalysisResults
from evaluation.constants import COLORS, CORR_METHODS_CORRESP

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

def create_auroc_barplot(
    results: List[pd.DataFrame],
    noise_levels: List[str],
    barplot_colors: Dict[str, str],
    strategies_dict: Dict,
    task: str,
    variation: str,
    dataset_name: str,
    decomp: str,
    output_path: Path,
    spatial: str = None,
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
    
    # Check if correlation-based colors file exists
    color_file_path = output_path / "color" / f"correlation_matrix_spearman_joint_noise_{dataset_name}_{task}_{variation}_dropout_pu_combined_methods_colors.json"
    
    correlation_colors = {}
    if color_file_path.exists():
        try:
            with open(color_file_path, 'r') as f:
                correlation_data = json.load(f)
                correlation_colors = {method: data.get('color', '#000000') for method, data in correlation_data.items()}
            print(f"Using correlation-based colors from: {color_file_path}")
        except Exception as e:
            print(f"Error loading correlation colors: {e}")
            correlation_colors = {}
    
    # Create mapping from display names to internal method names
    CORR_METHODS_CORRESP = {
        'Context-aware': {
            'equally-w. class avg.': 'class_mean_w_equal_weights',
            'imbalance-w. class avg.': 'class_mean_weighted_by_occurrence',
        },
        'Baseline': {
            'mean': 'mean',
        },
        'Threshold': {
            'ata 0.2': 'above_threshold_mean_0.2',
            'ata 0.3': 'above_threshold_mean_0.3',
            'ata 0.5': 'above_threshold_mean_0.5',
            'ata 0.7': 'above_threshold_mean_0.7',
            'ata 0.9': 'above_threshold_mean_0.9',
            'ata 0.95': 'above_threshold_mean_0.95',
        },
        'Quantile': {
            'aqa 0.3': 'above_quantile_mean_0.3',
            'aqa 0.5': 'above_quantile_mean_0.5',
            'aqa 0.7': 'above_quantile_mean_0.7',
            'aqa 0.9': 'above_quantile_mean_0.9',
            'aqa fg. ratio': 'above_quantile_mean_fg_ratio',
        },
        'Patch': {
            'plm 10': 'patch_aggregation_10',
            'plm 20': 'patch_aggregation_20',
            'plm 40': 'patch_aggregation_40',
            'plm 60': 'patch_aggregation_60',
            'plm 80': 'patch_aggregation_80',
            'plm 100': 'patch_aggregation_100',
            'plm 200': 'patch_aggregation_200',
        },
    }
    
    # Create reverse mapping for easier lookup
    display_to_internal = {}
    for category, methods in CORR_METHODS_CORRESP.items():
        for display_name, internal_name in methods.items():
            display_to_internal[display_name] = internal_name
    
    # Create figure with subplots in a single row
    fig, axes = plt.subplots(1, len(noise_levels), figsize=(20, 5))
    
    # Create a mapping from each method to its high-level category
    method_to_category = {
        method: category
        for category, methods in strategies_dict.items()
        for method in methods.keys()
    }
    
    # Function to get color for a method
    def get_method_color(method_name, category):
        # First try to find the method in correlation colors
        if correlation_colors:
            # Try direct match first
            if method_name in correlation_colors:
                return correlation_colors[method_name]
            
            # Try to find matching internal name
            method_lower = method_name.lower()
            for display_name, internal_name in display_to_internal.items():
                if display_name.lower() == method_lower or method_lower.replace(' ', '').replace('-', '').replace('.', '') == display_name.lower().replace(' ', '').replace('-', '').replace('.', ''):
                    if internal_name in correlation_colors:
                        return correlation_colors[internal_name]
            
            # Try partial matching for common patterns
            for internal_name, color in correlation_colors.items():
                if 'mean' in method_lower and 'mean' in internal_name:
                    return color
                elif 'threshold' in method_lower and 'threshold' in internal_name:
                    # Try to match threshold values
                    import re
                    method_match = re.search(r'(\d+\.?\d*)', method_name)
                    internal_match = re.search(r'(\d+\.?\d*)', internal_name)
                    if method_match and internal_match and method_match.group(1) == internal_match.group(1):
                        return color
                elif 'quantile' in method_lower and 'quantile' in internal_name:
                    # Try to match quantile values
                    import re
                    method_match = re.search(r'(\d+\.?\d*)', method_name)
                    internal_match = re.search(r'(\d+\.?\d*)', internal_name)
                    if method_match and internal_match and method_match.group(1) == internal_match.group(1):
                        return color
                elif 'patch' in method_lower and 'patch' in internal_name:
                    # Try to match patch values
                    import re
                    method_match = re.search(r'(\d+)', method_name)
                    internal_match = re.search(r'(\d+)', internal_name)
                    if method_match and internal_match and method_match.group(1) == internal_match.group(1):
                        return color
        
        # Fallback to original category-based colors
        return barplot_colors.get(category, '#000000')
    
    # Create plots for each noise level
    for idx, (noise_level, df) in enumerate(zip(noise_levels, results)):
        ax = axes[idx]
        
        # Get colors for each method
        colors = [get_method_color(method, method_to_category[method]) for method in df['Aggregator']]
        
        bars = ax.bar(
            df['Aggregator'],
            df['AUROC'],
            yerr=df['AUROC_std'],
            color=colors,
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
    
    # Create legend - use correlation colors if available, otherwise use original colors
    if correlation_colors:
        # Create legend based on categories but with correlation colors
        legend_elements = []
        for category in strategies_dict.keys():
            # Use the first method's color from this category as representative
            methods_in_category = [method for method in df['Aggregator'] if method_to_category[method] == category]
            if methods_in_category:
                representative_color = get_method_color(methods_in_category[0], category)
                legend_elements.append(Patch(facecolor=representative_color, label=category))
    else:
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
    file_name = f'ood_auroc_{task}_{dataset_name}_{variation}_{decomp}'
    if spatial:
        file_name += f'_{spatial}'
    output_file = output_path.joinpath(f'figures/{file_name}_barplot.png')
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

def create_single_auroc_barplot(
    results: pd.DataFrame,
    barplot_colors: Dict[str, str],
    strategies_dict: Dict,
    task: str,
    variation: str,
    dataset_name: str,
    decomp: str,
    output_path: Path,
    spatial: str = None,
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
    
    # Check if correlation-based colors file exists
    if dataset_name == 'ade20k': 
        color_file_path = output_path / "figures" / "colors" / f"correlation_matrix_spearman_joint_noise_{dataset_name}_deeplabv3_{task}_{variation}_dropout_pu_combined_method_colors.json"
    else:
        color_file_path = output_path / "figures" / "colors" / f"correlation_matrix_spearman_joint_noise_{dataset_name}_{task}_{variation}_dropout_pu_combined_method_colors.json"
    
    correlation_colors = {}
    print(f"Looking for color file at: {color_file_path}")
    if color_file_path.exists():
        print("Color file found!")
        try:
            with open(color_file_path, 'r') as f:
                correlation_data = json.load(f)
                # print(f"Loaded correlation data keys: {list(correlation_data.keys())}")
                # Filter out any non-dict entries and ensure we get the color field
                correlation_colors = {}
                for method, data in correlation_data.items():
                    if isinstance(data, dict) and 'color' in data:
                        color = data['color']
                        # Handle NaN colors by assigning gray
                        if color == '#000000' or color is None or (isinstance(color, float) and pd.isna(color)):
                            color = '#C0C0C0'  # Gray color for NaN/missing values
                        correlation_colors[method] = color
                    else:
                        print(f"Warning: Invalid data format for method {method}: {data}")
                # print(f"Final correlation colors: {correlation_colors}")
            print(f"Using correlation-based colors from: {color_file_path}")
        except Exception as e:
            print(f"Error loading correlation colors: {e}")
            correlation_colors = {}
    else:
        print("Color file not found, using default colors")
        
    # Create reverse mapping for easier lookup
    display_to_internal = {}
    for category, methods in CORR_METHODS_CORRESP.items():
        for display_name, internal_name in methods.items():
            display_to_internal[display_name] = internal_name
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    # Create a mapping from each method to its high-level category
    method_to_category = {
        method: category
        for category, methods in strategies_dict.items()
        for method in methods.keys()
    }
    
    # Function to get color for a method
    def get_method_color(method_name, category):
        # First try to find the method in correlation colors
        if correlation_colors:
            # Try direct match first (exact case-sensitive match)
            if method_name in correlation_colors:
                return correlation_colors[method_name]
            
            # Create a mapping from AUROC display names to internal JSON names
            auroc_to_json_mapping = {
                # Context-aware
                'Equally-w. class avg.': 'class_mean_w_equal_weights',
                'Imbalance-w. class avg.': 'class_mean_weighted_by_occurrence',
                
                # Baseline
                'Mean': 'mean',
                
                # Threshold
                'Threshold 0.3': 'above_threshold_mean_0.3',
                'Threshold 0.5': 'above_threshold_mean_0.5',
                'Threshold 0.7': 'above_threshold_mean_0.7',
                
                # Quantile (note: display names don't match the actual parameters used!)
                'Quantile 0.6': 'above_quantile_mean_0.5',  # Uses parameter 0.5
                'Quantile 0.75': 'above_quantile_mean_0.7', # Uses parameter 0.7
                'Quantile 0.9': 'above_quantile_mean_0.9',  # Uses parameter 0.9
                'Quantile fg. ratio': 'above_quantile_mean_fg_ratio',
                
                # Patch
                'Patch 10': 'patch_aggregation_10',
                'Patch 20': 'patch_aggregation_20',
                'Patch 50': 'patch_aggregation_40',  # Uses parameter 40, not 50!
            }
            
            # Try direct mapping
            if method_name in auroc_to_json_mapping:
                json_name = auroc_to_json_mapping[method_name]
                if json_name in correlation_colors:
                    return correlation_colors[json_name]
            
            # If no direct mapping found, try the old CORR_METHODS_CORRESP approach
            # but with case-insensitive matching
            method_lower = method_name.lower()
            for display_name, internal_name in display_to_internal.items():
                if (display_name.lower() == method_lower or 
                    method_lower.replace(' ', '').replace('-', '').replace('.', '') == 
                    display_name.lower().replace(' ', '').replace('-', '').replace('.', '')):
                    if internal_name in correlation_colors:
                        return correlation_colors[internal_name]
        
        # If we get here, something went wrong - we should have found a match
        print(f"Warning: Could not find color for method '{method_name}' in category '{category}'")
        print(f"Available correlation colors: {list(correlation_colors.keys()) if correlation_colors else 'None'}")
        
        # Fallback to original category-based colors (and handle RGB tuples)
        fallback_color = barplot_colors.get(category, '#C0C0C0')
        
        # Convert RGB tuple to hex if necessary
        if isinstance(fallback_color, tuple):
            if len(fallback_color) >= 3:
                # Convert from 0-1 range to 0-255 range if necessary
                if all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in fallback_color[:3]):
                    r, g, b = [int(x * 255) for x in fallback_color[:3]]
                else:
                    r, g, b = [int(x) for x in fallback_color[:3]]
                return f'#{r:02x}{g:02x}{b:02x}'
            else:
                return '#C0C0C0'
        
        return fallback_color
    
    # Get colors for each method
    colors = [get_method_color(method, method_to_category[method]) for method in results['Aggregator']]
        
    # Create bar plot
    bars = ax.bar(
        results['Aggregator'],
        results['AUROC'],
        yerr=results['AUROC_std'],
        color=colors,
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
    
    # Add method label inside the bar, rotated vertically
    for bar, label in zip(bars, results['Aggregator']):
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
    
    # Set labels and formatting
    ax.set_ylabel('AUROC' + r" $\uparrow$", fontsize=12)
    ax.set_ylim(0, 1)  # AUROC is between 0 and 1
    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(axis='y', which='major', labelsize=13)
    ax.set(xticklabels=[])
    ax.tick_params(bottom=False)
    
    # Create legend - use correlation colors if available, otherwise use original colors
    if correlation_colors:
        # Create legend based on categories but with correlation colors
        legend_elements = []
        for category in strategies_dict.keys():
            # Use the first method's color from this category as representative
            methods_in_category = [method for method in results['Aggregator'] if method_to_category[method] == category]
            if methods_in_category:
                representative_color = get_method_color(methods_in_category[0], category)
                legend_elements.append(Patch(facecolor=representative_color, label=category))
    else:
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
    file_name = f'ood_auroc_{task}_{dataset_name}_{variation}_{decomp}'
    if spatial:
        file_name += f'_{spatial}'
    output_file = output_path.joinpath(f'figures/{file_name}_barplot.png')
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
    dataset_name: str,
    decomp: str,
    output_path: Path,
    spatial: str = None,
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
    
    # Check if correlation-based colors file exists
    if dataset_name == 'ade20k': 
        color_file_path = output_path / "figures" / "colors" / f"correlation_matrix_spearman_joint_noise_{dataset_name}_deeplabv3_{task}_{variation}_dropout_pu_combined_method_colors.json"
    else:
        color_file_path = output_path / "figures" / "colors" / f"correlation_matrix_spearman_joint_noise_{dataset_name}_{task}_{variation}_dropout_pu_combined_method_colors.json"
    
    correlation_colors = {}
    print(f"Looking for color file at: {color_file_path}")
    if color_file_path.exists():
        print("Color file found!")
        try:
            with open(color_file_path, 'r') as f:
                correlation_data = json.load(f)
                # print(f"Loaded correlation data keys: {list(correlation_data.keys())}")
                # Filter out any non-dict entries and ensure we get the color field
                correlation_colors = {}
                for method, data in correlation_data.items():
                    if isinstance(data, dict) and 'color' in data:
                        color = data['color']
                        # Handle NaN colors by assigning gray
                        if color == '#000000' or color is None or (isinstance(color, float) and pd.isna(color)):
                            color = '#C0C0C0'  # Gray color for NaN/missing values
                        correlation_colors[method] = color
                    else:
                        print(f"Warning: Invalid data format for method {method}: {data}")
                # print(f"Final correlation colors: {correlation_colors}")
            print(f"Using correlation-based colors from: {color_file_path}")
        except Exception as e:
            print(f"Error loading correlation colors: {e}")
            correlation_colors = {}
    else:
        print("Color file not found, using default colors")
        
    # Create reverse mapping for easier lookup
    display_to_internal = {}
    for category, methods in CORR_METHODS_CORRESP.items():
        for display_name, internal_name in methods.items():
            display_to_internal[display_name] = internal_name
    
    # Create figure with subplots in a single row
    fig, axes = plt.subplots(1, len(noise_levels), figsize=(20, 5))
    
    # Create a mapping from each method to its high-level category
    method_to_category = {
        method: category
        for category, methods in strategies_dict.items()
        for method in methods.keys()
    }
    
    # Function to get color for a method
    def get_method_color(method_name, category):
        # First try to find the method in correlation colors
        if correlation_colors:
            # Try direct match first (exact case-sensitive match)
            if method_name in correlation_colors:
                return correlation_colors[method_name]
            
            # Create a mapping from AUROC display names to internal JSON names
            auroc_to_json_mapping = {
                # Context-aware
                'Equally-w. class avg.': 'class_mean_w_equal_weights',
                'Imbalance-w. class avg.': 'class_mean_weighted_by_occurrence',
                
                # Baseline
                'Mean': 'mean',
                
                # Threshold
                'Threshold 0.3': 'above_threshold_mean_0.3',
                'Threshold 0.5': 'above_threshold_mean_0.5',
                'Threshold 0.7': 'above_threshold_mean_0.7',
                
                # Quantile (note: display names don't match the actual parameters used!)
                'Quantile 0.6': 'above_quantile_mean_0.5',  # Uses parameter 0.5
                'Quantile 0.75': 'above_quantile_mean_0.7', # Uses parameter 0.7
                'Quantile 0.9': 'above_quantile_mean_0.9',  # Uses parameter 0.9
                'Quantile fg. ratio': 'above_quantile_mean_fg_ratio',
                
                # Patch
                'Patch 10': 'patch_aggregation_10',
                'Patch 20': 'patch_aggregation_20',
                'Patch 50': 'patch_aggregation_40',  # Uses parameter 40, not 50!
            }
            
            # Try direct mapping
            if method_name in auroc_to_json_mapping:
                json_name = auroc_to_json_mapping[method_name]
                if json_name in correlation_colors:
                    return correlation_colors[json_name]
            
            # If no direct mapping found, try the old CORR_METHODS_CORRESP approach
            # but with case-insensitive matching
            method_lower = method_name.lower()
            for display_name, internal_name in display_to_internal.items():
                if (display_name.lower() == method_lower or 
                    method_lower.replace(' ', '').replace('-', '').replace('.', '') == 
                    display_name.lower().replace(' ', '').replace('-', '').replace('.', '')):
                    if internal_name in correlation_colors:
                        return correlation_colors[internal_name]
        
        # If we get here, something went wrong - we should have found a match
        print(f"Warning: Could not find color for method '{method_name}' in category '{category}'")
        print(f"Available correlation colors: {list(correlation_colors.keys()) if correlation_colors else 'None'}")
        
        # Fallback to original category-based colors (and handle RGB tuples)
        fallback_color = barplot_colors.get(category, '#C0C0C0')
        
        # Convert RGB tuple to hex if necessary
        if isinstance(fallback_color, tuple):
            if len(fallback_color) >= 3:
                # Convert from 0-1 range to 0-255 range if necessary
                if all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in fallback_color[:3]):
                    r, g, b = [int(x * 255) for x in fallback_color[:3]]
                else:
                    r, g, b = [int(x) for x in fallback_color[:3]]
                return f'#{r:02x}{g:02x}{b:02x}'
            else:
                return '#C0C0C0'
        
        return fallback_color
    
    # Create plots for each noise level
    for idx, (noise_level, df) in enumerate(zip(noise_levels, results)):
        ax = axes[idx]
        
        # Get colors for each method
        colors = [get_method_color(method, method_to_category[method]) for method in df['Aggregator']]
        
        bars = ax.bar(
            df['Aggregator'],
            df['AUROC'],
            yerr=df['AUROC_std'],
            color=colors,
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
    
    # Create legend - use correlation colors if available, otherwise use original colors
    if correlation_colors:
        # Create legend based on categories but with correlation colors
        legend_elements = []
        for category in strategies_dict.keys():
            # Use the first method's color from this category as representative
            methods_in_category = [method for method in df['Aggregator'] if method_to_category[method] == category]
            if methods_in_category:
                representative_color = get_method_color(methods_in_category[0], category)
                legend_elements.append(Patch(facecolor=representative_color, label=category))
    else:
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
    file_name = f'ood_auroc_{task}_{dataset_name}_{variation}_{decomp}'
    if spatial:
        file_name += f'_{spatial}'
    output_file = output_path.joinpath(f'figures/{file_name}_barplot.png')
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