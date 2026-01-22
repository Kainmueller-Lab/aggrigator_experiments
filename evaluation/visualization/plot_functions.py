import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pathlib import Path
from typing import Dict, List, Tuple, Optional
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

# def create_auroc_barplot(
#     results: List[pd.DataFrame],
#     noise_levels: List[str],
#     barplot_colors: Dict[str, str],
#     strategies_dict: Dict,
#     task: str,
#     variation: str,
#     dataset_name: str,
#     decomp: str,
#     output_path: Path,
#     spatial: str = None,
# ) -> None:
#     """
#     Create comparative bar plots of image-level AUROC values.
#     Parameters
#     ----------
#     results : List[pd.DataFrame]
#         List of DataFrames with AUROC results for each noise level
#     noise_levels : List[str]
#         List of noise levels
#     barplot_colors : Dict[str, str]
#         Dictionary mapping categories to colors
#     strategies_dict : Dict
#         Dictionary of strategies by category
#     task : str
#         Task type ('instance' or 'semantic')
#     variation : str
#         Variation type
#     output_path : Path
#         Path to save the output figure
#     """
#     # Create colors path
#     if results['Aggregator'].str.contains('gmm_normalized').any():
#         pre_color_path = output_path / "figures" / "joint_correlation" / "colors"
#     else:
#         pre_color_path = output_path / "figures" / "colors"
    
#     # Check if correlation-based colors file exists
#     if dataset_name == 'ade20k': 
#         color_file_path = pre_color_path / f"correlation_matrix_spearman_joint_noise_{dataset_name}_deeplabv3_{task}_{variation}_dropout_pu_combined_method_colors.json"
#     else:
#         color_file_path = pre_color_path / f"correlation_matrix_spearman_joint_noise_{dataset_name}_{task}_{variation}_dropout_pu_combined_method_colors.json"
    
#     correlation_colors = {}
#     print(f"Looking for color file at: {color_file_path}")
#     if color_file_path.exists():
#         print("Color file found!")
#         try:
#             with open(color_file_path, 'r') as f:
#                 correlation_data = json.load(f)
#                 # print(f"Loaded correlation data keys: {list(correlation_data.keys())}")
#                 # Filter out any non-dict entries and ensure we get the color field
#                 correlation_colors = {}
#                 for method, data in correlation_data.items():
#                     if isinstance(data, dict) and 'color' in data:
#                         color = data['color']
#                         # Handle NaN colors by assigning gray
#                         if color == '#000000' or color is None or (isinstance(color, float) and pd.isna(color)):
#                             color = '#C0C0C0'  # Gray color for NaN/missing values
#                         correlation_colors[method] = color
#                     else:
#                         print(f"Warning: Invalid data format for method {method}: {data}")
#                 # print(f"Final correlation colors: {correlation_colors}")
#             print(f"Using correlation-based colors from: {color_file_path}")
#         except Exception as e:
#             print(f"Error loading correlation colors: {e}")
#             correlation_colors = {}
#     else:
#         print("Color file not found, using default colors")
        
#     # Create reverse mapping for easier lookup
#     display_to_internal = {}
#     for category, methods in CORR_METHODS_CORRESP.items():
#         for display_name, internal_name in methods.items():
#             display_to_internal[display_name] = internal_name
    
#     # Create figure with subplots in a single row
#     fig, axes = plt.subplots(1, len(noise_levels), figsize=(20, 5))
    
#      # Create a mapping from each method to its high-level category
#     method_to_category = {
#         method: category
#         for category, methods in strategies_dict.items()
#         for method in methods.keys()
#     }
    
#     # Function to get color for a method
#     def get_method_color(method_name, category):
#         # First try to find the method in correlation colors
#         if correlation_colors:
#             # Try direct match first (exact case-sensitive match)
#             if method_name in correlation_colors:
#                 return correlation_colors[method_name]
            
#             # Create a mapping from AUROC display names to internal JSON names
#             auroc_to_json_mapping = {
#                 # Context-aware
#                 'Equally-w. class avg.': 'class_mean_w_equal_weights',
#                 'Imbalance-w. class avg.': 'class_mean_weighted_by_occurrence',
                
#                 # Baseline
#                 'Mean': 'mean',
                
#                 # Threshold
#                 'Threshold 0.3': 'above_threshold_mean_0.3',
#                 'Threshold 0.5': 'above_threshold_mean_0.5',
#                 'Threshold 0.7': 'above_threshold_mean_0.7',
                
#                 # Quantile (note: display names don't match the actual parameters used!)
#                 'Quantile 0.6': 'above_quantile_mean_0.5',  # Uses parameter 0.5
#                 'Quantile 0.75': 'above_quantile_mean_0.7', # Uses parameter 0.7
#                 'Quantile 0.9': 'above_quantile_mean_0.9',  # Uses parameter 0.9
#                 'Quantile fg. ratio': 'above_quantile_mean_fg_ratio',
                
#                 # Patch
#                 'Patch 10': 'patch_aggregation_10',
#                 'Patch 20': 'patch_aggregation_20',
#                 'Patch 50': 'patch_aggregation_40',  # Uses parameter 40, not 50!
#             }
            
#             # Try direct mapping
#             if method_name in auroc_to_json_mapping:
#                 json_name = auroc_to_json_mapping[method_name]
#                 if json_name in correlation_colors:
#                     return correlation_colors[json_name]
            
#             # If no direct mapping found, try the old CORR_METHODS_CORRESP approach
#             # but with case-insensitive matching
#             method_lower = method_name.lower()
#             for display_name, internal_name in display_to_internal.items():
#                 if (display_name.lower() == method_lower or 
#                     method_lower.replace(' ', '').replace('-', '').replace('.', '') == 
#                     display_name.lower().replace(' ', '').replace('-', '').replace('.', '')):
#                     if internal_name in correlation_colors:
#                         return correlation_colors[internal_name]
        
#         # If we get here, something went wrong - we should have found a match
#         print(f"Warning: Could not find color for method '{method_name}' in category '{category}'")
#         print(f"Available correlation colors: {list(correlation_colors.keys()) if correlation_colors else 'None'}")
        
#         # Fallback to original category-based colors (and handle RGB tuples)
#         fallback_color = barplot_colors.get(category, '#C0C0C0')
        
#         # Convert RGB tuple to hex if necessary
#         if isinstance(fallback_color, tuple):
#             if len(fallback_color) >= 3:
#                 # Convert from 0-1 range to 0-255 range if necessary
#                 if all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in fallback_color[:3]):
#                     r, g, b = [int(x * 255) for x in fallback_color[:3]]
#                 else:
#                     r, g, b = [int(x) for x in fallback_color[:3]]
#                 return f'#{r:02x}{g:02x}{b:02x}'
#             else:
#                 return '#C0C0C0'
        
#         return fallback_color
    
#     # Create plots for each noise level
#     for idx, (noise_level, df) in enumerate(zip(noise_levels, results)):
#         ax = axes[idx]
        
#         # Get colors for each method
#         colors = [get_method_color(method, method_to_category[method]) for method in df['Aggregator']]
        
#         bars = ax.bar(
#             df['Aggregator'],
#             df['AUROC'],
#             yerr=df['AUROC_std'],
#             color=colors,
#             capsize=4,
#             zorder=3,
#         )
        
#         # Add AUROC values on top of bars
#         for bar in bars:
#             height = bar.get_height()
#             ax.annotate(
#                 f'{height:.3f}',
#                 xy=(bar.get_x() + bar.get_width()/2, height),
#                 xytext=(0, 3),  # 3 points vertical offset
#                 textcoords="offset points",
#                 ha='center', va='bottom'
#             )
        
#         # Add method label inside the bar, rotated horizontally
#         for bar, label in zip(bars, df['Aggregator']):
#             y_offset = 0.005 * 2 * bar.get_height()  # Adjust offset as needed
#             ax.text(
#                 bar.get_x() + bar.get_width() / 2,
#                 y_offset,
#                 label,
#                 ha="center",
#                 va="bottom",
#                 rotation="vertical",
#                 fontsize=15,
#                 zorder=4,
#             )
        
#         ax.set_title(f'Noise Level: {noise_level}')
#         ax.set_ylabel('AUROC' + r" $\uparrow$", fontsize=12)
#         ax.set_ylim(0, 1)  # AUROC is between 0 and 1
#         ax.spines[['right', 'top']].set_visible(False)
#         ax.tick_params(axis='y', which='major', labelsize=13)
#         ax.set(xticklabels=[])
#         ax.tick_params(bottom=False)
    
#     # Create legend - use correlation colors if available, otherwise use original colors
#     if correlation_colors:
#         # Create legend based on categories but with correlation colors
#         legend_elements = []
#         for category in strategies_dict.keys():
#             # Use the first method's color from this category as representative
#             methods_in_category = [method for method in df['Aggregator'] if method_to_category[method] == category]
#             if methods_in_category:
#                 representative_color = get_method_color(methods_in_category[0], category)
#                 legend_elements.append(Patch(facecolor=representative_color, label=category))
#     else:
#         legend_elements = [
#             Patch(facecolor=v, label=k)
#             for k, v in barplot_colors.items()
#         ]
    
#     fig.legend(
#         handles=legend_elements,
#         loc='upper center',
#         bbox_to_anchor=(0.5, 0.05),
#         fancybox=True,
#         shadow=True,
#         ncol=3
#     )
    
#     # Add title
#     plt.suptitle(
#         f'OOD correctness measured by the AUROC w.r.t. model confidence correctness.\n'
#         f'Task: {task}, Variation: {variation}',
#         fontsize=16
#     )
#     plt.tight_layout(rect=[0, 0, 1, 0.90])
    
#     # Ensure output directory exists
#     file_name = f'ood_auroc_{task}_{dataset_name}_{variation}_{decomp}'
#     if spatial:
#         file_name += f'_{spatial}'
#     output_file = output_path.joinpath(f'figures/auroc_gmm/{file_name}_barplot.png')
#     output_file.parent.mkdir(exist_ok=True, parents=True)
    
#     # Save the plot
#     plt.savefig(output_file, dpi=300, bbox_inches='tight')
#     print(f"Plot saved to: {output_file}")

def create_auroc_barplot(
    results: List[pd.DataFrame],
    noise_levels: List[str],
    barplot_colors: Dict[str, str],
    strategies_dict: Dict,
    task: str,
    variation: str,
    dataset_name: str,
    uq_method: str,
    decomp: str,
    output_path: Path,
    spatial: str = None,
) -> None:
    """
    Create comparative bar plots of image-level AUROC values.
    """
    # Create figure with subplots in a single row
    fig, axes = plt.subplots(1, len(noise_levels), figsize=(20, 5), sharey=True)
    if len(noise_levels) == 1:
        axes = [axes] # Make sure axes is always iterable

    # --- DYNAMIC STRATEGY MODIFICATION ---
    # Check if GMM score exists in the first result dataframe to decide on modifications
    strategies_dict_local = strategies_dict.copy()
    barplot_colors_local = barplot_colors.copy()
    
    # Check if GMM score exists in the results to set the correct path
    if not results[0].empty and 'GMM' in results[0]['Aggregator'].values:
        print("GMM score found. Using joint_correlation color path and adding 'Spatial' category.")
        pre_color_path = output_path / "figures" / "joint_correlation" / "colors"
        
        # Dynamically add the new category for the legend
        strategies_dict_local['Spatial GMM'] = {'GMM': (None, None)}
        barplot_colors_local['Spatial GMM'] = '#C0C0C0' # Default gray, will be replaced by JSON color if found
    else:
        print("GMM score not found. Using standard color path.")
        pre_color_path = output_path / "figures" / "colors"
    # --- END DYNAMIC MODIFICATION ---

    # Get the path to the correlation color file
    color_file_path = _get_color_file_path(pre_color_path, dataset_name, task, variation)
    correlation_colors = _load_correlation_colors(color_file_path)

    # Create a mapping from each method to its high-level category
    method_to_category = _create_method_category_mapping(strategies_dict_local)

    # Create plots for each noise level
    for idx, (noise_level, df) in enumerate(zip(noise_levels, results)):
        ax = axes[idx]
        if df.empty:
            ax.set_title(f'Noise Level: {noise_level}\n(No data)')
            continue

        # Get colors for each method
        colors = [_get_method_color(method, method_to_category, 
                                   correlation_colors, barplot_colors_local) 
                  for method in df['Aggregator']]
        
        # Create bar plot
        bars = ax.bar(
            df['Aggregator'], df['AUROC'], yerr=df.get('AUROC_std'),
            color=colors, capsize=4, zorder=3,
        )
        
        # Add formatting
        _format_bars(ax, bars, df)
        _format_axes(ax)
        ax.set_title(f'Noise Level: {noise_level}')
        
    # Add a single, shared legend
    _add_legend(fig, strategies_dict_local, method_to_category, results[0], 
                correlation_colors, barplot_colors_local, legend_ncol=len(strategies_dict_local))
    
    # Add main title
    plt.suptitle(
        f'OOD correctness measured by the AUROC w.r.t. model confidence correctness.\n'
        f'Task: {task}, Variation: {variation}',
        fontsize=16
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.90]) # Adjust rect to make space for legend
    
    # Save the plot
    _save_plot(fig, output_path, task, dataset_name, variation, uq_method, decomp, spatial)
    plt.close()

# --- Helper Functions  ---

def _get_color_file_path(pre_color_path: Path, dataset_name: str, task: str, variation: str) -> Path:
    """Constructs the path to the correlation color file."""
    # This logic assumes GMM is always run with dropout, which seems to be the case.    
    if dataset_name == 'ade20k': 
        return pre_color_path / f"correlation_matrix_spearman_joint_noise_{dataset_name}_deeplabv3_{task}_{variation}_dropout_pu_combined_method_colors.json"
    else:
        return pre_color_path / f"correlation_matrix_spearman_joint_noise_{dataset_name}_{task}_{variation}_dropout_pu_combined_method_colors.json"
    
def create_single_auroc_barplot(
    results: pd.DataFrame,
    barplot_colors: Dict[str, str],
    strategies_dict: Dict,
    task: str,
    variation: str,
    dataset_name: str,
    uq_method: str,
    decomp: str,
    output_path: Path,
    spatial: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 5),
) -> None:
    """
    Create a single bar plot of image-level AUROC values.
    
    Parameters
    ----------
    results : pd.DataFrame
        DataFrame with AUROC results containing columns: 'Aggregator', 'AUROC', 'AUROC_std'
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
    uq_method : str
        Uncertainty quantification method to evaluate (e.g. 'dropout')
    decomp : str
        Uncertainty component tested ('pu', 'eu' or 'au')
    output_path : Path
        Path to save the output figure
    spatial : Optional[str], default=None
        Spatial component identifier
    figsize : Tuple[int, int], default=(6, 5)
        Figure size as (width, height)
    """
    # Build color file path
    # if results['Aggregator'].str.contains('gmm_normalized').any():
    #     pre_color_path = output_path / "figures" / "joint_correlation" / "colors"
    # else:
    #     pre_color_path = output_path / "figures" / "colors"
    
    # if dataset_name == 'ade20k':
    #     color_file_path = (pre_color_path / 
    #                       f"correlation_matrix_spearman_joint_noise_{dataset_name}_deeplabv3_{task}_{variation}_dropout_pu_combined_method_colors.json")
    # else:
    #     color_file_path = (pre_color_path / 
    #                       f"correlation_matrix_spearman_joint_noise_{dataset_name}_{task}_{variation}_dropout_pu_combined_method_colors.json")
    
    # --- DYNAMIC PATH AND STRATEGY LOGIC (mirrored from multi-plot function) ---
    strategies_dict_local = strategies_dict.copy()
    barplot_colors_local = barplot_colors.copy()
    
    gmm_methods_present = not results.empty and results['Aggregator'].str.startswith('GMM').any()

    if gmm_methods_present:
        print("GMM scores found. Using joint_correlation color path and adding 'Spatial' category.")
        pre_color_path = output_path / "figures" / "joint_correlation" / "colors"
        
        # Add a single 'Spatial' category containing all GMM variants.
        # This ensures they are grouped correctly for the legend.
        strategies_dict_local['Spatial'] = {
            'GMM': (None, None),
            'GMM_pixel': (None, None),
            'GMM_spatial': (None, None)
        }
        
        # Assign a generic color for the 'Spatial' category itself (used for the legend).
        # The specific bar colors for each method will be handled by _get_method_color.
        barplot_colors_local['Spatial'] = '#7f7f7f'  # A neutral gray
    else:
        print("GMM scores not found. Using standard color path.")
        pre_color_path = output_path / "figures" / "colors"

    # --- END DYNAMIC LOGIC ---

    color_file_path = _get_color_file_path(pre_color_path, dataset_name, task, variation)
    
    logger.info(f"Looking for color file at: {color_file_path}")
 
    # Load correlation-based colors
    correlation_colors = _load_correlation_colors(color_file_path, output_path)
    
    # Create method to category mapping
    method_to_category = _create_method_category_mapping(strategies_dict_local)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Get colors for each method
    colors = [_get_method_color(method, method_to_category[method], 
                               correlation_colors, barplot_colors_local) 
              for method in results['Aggregator']]
        
    # Create bar plot
    bars = ax.bar(
        results['Aggregator'],
        results['AUROC'],
        yerr=results['AUROC_std'],
        color=colors,
        capsize=4,
        zorder=3,
    )
    
    # Add formatting
    _format_bars(ax, bars, results)
    _format_axes(ax)
    
    # Add legend
    _add_legend(ax, strategies_dict_local, method_to_category, results, 
                correlation_colors, barplot_colors_local)
    # Add title
    _add_title(ax, task, variation)
    # Save plot
    _save_plot(fig, output_path, task, dataset_name, variation, uq_method, decomp, spatial)
    
    plt.close()  # Close figure to free memory

def _load_correlation_colors(color_file_path: Path, output_path: Path) -> Dict[str, str]:
    """Load correlation-based colors from file."""   
    if not color_file_path.exists():
        logger.info("Color file not found, using default colors")
        return {}
    
    try:
        with open(color_file_path, 'r') as f:
            correlation_data = json.load(f)
        
        correlation_colors = {}
        for method, data in correlation_data.items():
            if isinstance(data, dict) and 'color' in data:
                color = data['color']
                # Handle NaN colors by assigning gray
                if color == '#000000' or color is None or (isinstance(color, float) and pd.isna(color)):
                    color = '#C0C0C0'  # Gray color for NaN/missing values
                correlation_colors[method] = color
            else:
                logger.warning(f"Invalid data format for method {method}: {data}")
        
        logger.info(f"Using correlation-based colors from: {color_file_path}")
        return correlation_colors
        
    except Exception as e:
        logger.error(f"Error loading correlation colors: {e}")
        return {}

def _create_method_category_mapping(strategies_dict: Dict) -> Dict[str, str]:
    """Create a mapping from each method to its high-level category."""
    return {
        method: category
        for category, methods in strategies_dict.items()
        for method in methods.keys()
    }

def _get_method_color(method_name: str, category: str, correlation_colors: Dict[str, str], 
                     barplot_colors: Dict[str, str]) -> str:
    """Get color for a method, preferring correlation colors over category colors."""
    
    # This ensures they always get a specific pink color.
    # We use distinct shades of pink to differentiate them.
    if method_name == 'GMM_pixel':
        return '#FF69B4'  # Hot Pink, similar to selective risk plots
    if method_name == 'GMM_spatial':
        return '#C71585'  # Medium Violet Red, a darker pink
    
    if not correlation_colors:
        return _convert_color_to_hex(barplot_colors.get(category, '#C0C0C0'))
    
    # Try direct match first
    if method_name in correlation_colors:
        return correlation_colors[method_name]
    
    # Try mapping through AUROC display names
    color = _try_auroc_mapping(method_name, correlation_colors)
    if color:
        return color
        
    # Fallback to category color
    logger.warning(f"Could not find color for method '{method_name}' in category '{category}'")
    return _convert_color_to_hex(barplot_colors.get(category, '#C0C0C0'))

def _convert_color_to_hex(color) -> str:
    """Convert color to hex format."""
    if isinstance(color, tuple):
        if len(color) >= 3:
            # Convert from 0-1 range to 0-255 range if necessary
            if all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in color[:3]):
                r, g, b = [int(x * 255) for x in color[:3]]
            else:
                r, g, b = [int(x) for x in color[:3]]
            return f'#{r:02x}{g:02x}{b:02x}'
        else:
            return '#C0C0C0'
    return color

def _format_bars(ax, bars, results: pd.DataFrame, show_values: bool = True) -> None:
    """Format bars with values and labels."""
    if show_values:
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
    
    # Add method labels inside bars
    for bar, label in zip(bars, results['Aggregator']):
        y_offset = 0.005 * 2 * bar.get_height()
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

def _format_axes(ax) -> None:
    """Format axes appearance."""
    ax.set_ylabel('AUROC' + r" $\uparrow$", fontsize=12)
    ax.set_ylim(0, 1)  # AUROC is between 0 and 1 (?)
    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(axis='y', which='major', labelsize=13)
    ax.set(xticklabels=[])
    ax.tick_params(bottom=False)

def _try_auroc_mapping(method_name: str, correlation_colors: Dict[str, str]) -> Optional[str]:
    """Try to map method name through AUROC display names."""
    auroc_to_json_mapping = {
        'GMM Normalized': 'gmm_normalized_score',
        'Equally-w. class avg.': 'class_mean_w_equal_weights',
        'Imbalance-w. class avg.': 'class_mean_weighted_by_occurrence',
        'Mean': 'mean',
        'Threshold 0.3': 'above_threshold_mean_0.3',
        'Threshold 0.5': 'above_threshold_mean_0.5',
        'Threshold 0.7': 'above_threshold_mean_0.7',
        'Quantile 0.6': 'above_quantile_mean_0.5',
        'Quantile 0.75': 'above_quantile_mean_0.7',
        'Quantile 0.9': 'above_quantile_mean_0.9',
        'Quantile fg. ratio': 'above_quantile_mean_fg_ratio',
        'Patch 10': 'patch_aggregation_10',
        'Patch 20': 'patch_aggregation_20',
        'Patch 50': 'patch_aggregation_40',
    }
    
    if method_name in auroc_to_json_mapping:
        json_name = auroc_to_json_mapping[method_name]
        if json_name in correlation_colors:
            return correlation_colors[json_name]
    
    return None

def _add_legend(ax, strategies_dict: Dict, method_to_category: Dict, 
               results: pd.DataFrame, correlation_colors: Dict[str, str], 
               barplot_colors: Dict[str, str], legend_ncol: int = 3) -> None:
    """Add legend to the plot."""
    legend_elements = []
    
    for category in strategies_dict.keys():
        # Use the first method's color from this category as representative
        methods_in_category = [method for method in results['Aggregator'] 
                              if method_to_category[method] == category]
        
        if methods_in_category:
            representative_color = _get_method_color(
                methods_in_category[0], category, correlation_colors, barplot_colors
            )
            legend_elements.append(Patch(facecolor=representative_color, label=category))
    
    # Fallback to original colors if no methods found
    if not legend_elements:
        legend_elements = [
            Patch(facecolor=_convert_color_to_hex(v), label=k)
            for k, v in barplot_colors.items()
        ]
    
    ax.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.025),
        fancybox=True,
        shadow=True,
        ncol=legend_ncol
    )
    
def _add_title(ax, task: str, variation: str) -> None:
    """Add title to the plot."""
    title_text = (
        f'OOD correctness measured by the AUROC w.r.t. model confidence correctness.\n'
        f'Task: {task}, Variation: {variation}'
    )
    ax.set_title(title_text, fontsize=16, pad=20)
    
def _save_plot(fig, output_path: Path, task: str, dataset_name: str, variation: str, 
               uq_method: str, decomp: str, spatial: Optional[str]) -> None:
    """Save the plot to file."""
    file_name = f'ood_auroc_{task}_{dataset_name}_{variation}_{uq_method}_{decomp}'
    if spatial:
        file_name += f'_{spatial}'
    
    output_file = output_path / f'figures/auroc_gmm/{file_name}_barplot.png'
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to: {output_file}")

def create_selective_risks_coverage_plot(
        method_names: List[str],
        aurc_res: AnalysisResults,
        output_path: Path, 
        args: argparse.Namespace,
        ood : bool,
    ) -> None:
    """
    Create and save AURC plot.
    
    Args:
        method_names: List of method names
        augrc_res: Analysis results containing AURC data
        output_path: Path to save output
        args: Command line arguments
        ood: bool. to select between id and ood
    """
    # Define boolean to treat simulatenous evaluation on id and ood
    return_one_only = True if args.data_mod != 'id_ood' else False
    
    # Plot mean results
    x = aurc_res.coverages.flatten() # Flatten to 1D for plotting
    y = aurc_res.mean_selective_risks # Shape: [coverage points, num_strategies]
    y_std = aurc_res.std_selective_risks # Shape: same as y
    
    # Prepare data dictionary for CSV export
    data_dict = {"Coverage": x[::-1]} # Reverse to match plotting order
    
    # Define method categories for styling
    method_categories = ["Threshold", "Patch", "Quantile", "Quantile fg."]
    first_occurrence = {cat: True for cat in method_categories}
    
    # Build color file path
    color_variation = 'none' if not args.variation else args.variation
    
    if ood and return_one_only:
        color_noise = '1_00'
    elif not ood and return_one_only:
        color_noise = '0_00'
    else:
        color_noise = 'combined'
        
    if args.dataset_name == 'ade20k':
        color_file_path = (output_path / "figures" / "joint_correlation" / "colors" / 
                          f"correlation_matrix_spearman_joint_noise_{args.dataset_name}_deeplabv3_{args.task}_{color_variation}_dropout_pu_{color_noise}_method_colors.json")
    else:
        color_file_path = (output_path / "figures" / "joint_correlation" / "colors" / 
                          f"correlation_matrix_spearman_joint_noise_{args.dataset_name}_{args.task}_{color_variation}_dropout_pu_{color_noise}_method_colors.json")
    
    # Load correlation-based colors
    correlation_colors = _load_correlation_colors(color_file_path, output_path)

    # Plot each method
    for j, method_name in enumerate(method_names):
        # Add data to CSV export dictionary
        data_dict[f"{method_name} (Mean Risk)"] = y[:, j][::-1]
        data_dict[f"{method_name} (Std Dev)"] = y_std[:, j][::-1]
        
        # Get color and styling
        color, linestyle, alpha, linewidth, alpha_fill_current = _get_method_styling(
            method_name, j, method_categories, first_occurrence, 
            correlation_colors, 0.2, 0.1
        )
        
        # Plot line
        plt.plot(x[::-1], y[:, j][::-1], 
                 label=f"{method_names[j]} (AURC: {aurc_res.mean_aurc[j]:.4f})",
                 linewidth=linewidth, color=color, linestyle=linestyle, alpha=alpha)
        
        # Add shaded area (mean ± std)
        plt.fill_between(x[::-1], 
                        (y[:, j] - y_std[:, j])[::-1],
                        (y[:, j] + y_std[:, j])[::-1],
                        color=color, alpha=alpha_fill_current)
    
    # Save data to CSV
    _save_aurc_data(data_dict, output_path, args, ood, return_one_only)
    
    # Format and save plot
    _format_aurc_plot()
    _save_aurc_plot(output_path, args, ood, return_one_only)
    
    # Reconstruct the sorted dataframe to pass to the reporting function
    report_df = pd.DataFrame({
        'Aggregator': method_names, # These are already sorted
        'AURC': aurc_res.mean_aurc,
        'AURC_std': aurc_res.std_aurc,
        'EAURC': aurc_res.mean_eaurc,
        'EAURC_std': aurc_res.std_eaurc
    })
    
    # Create and save barplot
    create_metric_reports(report_df, output_path, args, ood, return_one_only, correlation_colors)
    
def _get_method_styling(method_name: str, method_index: int, method_categories: List[str], 
                       first_occurrence: Dict[str, bool], correlation_colors: Dict[str, str],
                       alpha_fill: float, alpha_fill_secondary: float) -> tuple:
    """Get styling parameters for a method."""
    # Default fallback colors (you may want to define these based on your original COLORS list)
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Get color from correlation colors or fallback
    color = _get_aurc_method_color(method_name, correlation_colors, 
                                  default_colors[method_index % len(default_colors)])
    
    # Default styling
    linestyle = '-'
    alpha = 1.0
    linewidth = 2
    alpha_fill_current = alpha_fill
    
    # Special handling for Mean method
    if method_name.startswith("Mean"):
        color = 'gray'
        linewidth = 2
        return color, linestyle, alpha, linewidth, alpha_fill_current

    if method_name.startswith("GMM"):
        linestyle = ':'  # Use a dotted line to make GMM methods stand out
        linewidth = 2  # Slightly thicker to be visible
        # The color will still be loaded from correlation_colors if available
        return color, linestyle, alpha, linewidth, alpha_fill_current
    
    # Check if the method belongs to a category for special styling
    for cat in method_categories:
        if method_name.startswith(cat):
            if cat == "Quantile fg.":
                linestyle = '-'
                break  # Do not modify style
            if first_occurrence[cat]:
                first_occurrence[cat] = False  # Mark first as used
            else:
                linestyle = '--'  # Dashed line for subsequent ones
                linewidth = 1  # Make it thinner
                alpha = 0.5  # Make it more transparent
                alpha_fill_current = alpha_fill_secondary
            break  # Exit loop once category is found
    
    return color, linestyle, alpha, linewidth, alpha_fill_current

def _get_aurc_method_color(method_name: str, correlation_colors: Dict[str, str], 
                          default_color: str) -> str:
    """Get color for AURC method, mapping from display name to internal name."""
    if not correlation_colors:
        return default_color
    
    # Try direct match first
    if method_name in correlation_colors:
        return correlation_colors[method_name]
    
    # AUROC display name to internal name mapping (same as before)
    auroc_to_json_mapping = {
        'Equally-w. class avg.': 'class_mean_w_equal_weights',
        'Imbalance-w. class avg.': 'class_mean_weighted_by_occurrence',
        'Mean': 'mean',
        'Threshold 0.3': 'above_threshold_mean_0.3',
        'Threshold 0.5': 'above_threshold_mean_0.5',
        'Threshold 0.7': 'above_threshold_mean_0.7',
        'Quantile 0.6': 'above_quantile_mean_0.5',
        'Quantile 0.75': 'above_quantile_mean_0.7',
        'Quantile 0.9': 'above_quantile_mean_0.9',
        'Quantile fg. ratio': 'above_quantile_mean_fg_ratio',
        'Patch 10': 'patch_aggregation_10',
        'Patch 20': 'patch_aggregation_20',
        'Patch 50': 'patch_aggregation_40',
    }
    
    # Try mapping through AUROC display names
    if method_name in auroc_to_json_mapping:
        json_name = auroc_to_json_mapping[method_name]
        if json_name in correlation_colors:
            return correlation_colors[json_name]
    
    # Try fuzzy matching with normalized strings
    method_normalized = _normalize_aurc_string(method_name)
    for internal_name, color in correlation_colors.items():
        if method_normalized in _normalize_aurc_string(internal_name):
            return color
    
    # Try partial matching for method categories
    for display_name, internal_name in auroc_to_json_mapping.items():
        if _normalize_aurc_string(display_name) == method_normalized:
            if internal_name in correlation_colors:
                return correlation_colors[internal_name]
    
    logger.warning(f"Could not find color for AURC method '{method_name}', using default")
    return default_color

def _normalize_aurc_string(s: str) -> str:
    """Normalize string for fuzzy matching in AURC context."""
    return s.lower().replace(' ', '').replace('-', '').replace('.', '').replace('_', '')

def _save_aurc_data(data_dict: Dict, output_path: Path, args: argparse.Namespace, ood: bool, return_one_only: bool) -> None:
    """Save AURC data to CSV file."""
    df = pd.DataFrame(data_dict)
    
    # Define output file paths
    if ood and return_one_only:
        data_mod = 'ood'
    elif not ood and return_one_only:
        data_mod = 'id'
    else:
        data_mod = 'id_ood'
    
    uq_methods = [uq.strip() for uq in args.uq_methods.split(',')] #Temporary placeholder to then adjust the functions in view of multiple uncertainty methods used 
    csv_file = output_path.joinpath(
        f'tables/aurc_{data_mod}/aurc_data_{args.aggregator_type}_aggr_{uq_methods[0]}_{args.task}_{args.variation}_{data_mod}.csv'
    )
    
    # Ensure directory exists
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists to handle headers
    file_empty = not csv_file.exists() or csv_file.stat().st_size == 0
    df.to_csv(csv_file, mode='a', index=False, header=file_empty)
    print(f"Data saved to: {csv_file}")
    logger.info(f"Data saved to: {csv_file}")
    
def _format_aurc_plot(legend_ncol: int = 5) -> None:
    """Format the AURC plot with labels, legend, and styling."""
    plt.xlabel("Coverage")
    plt.ylabel("Selective Risks")
    
    # Get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # Reorder manually: Mean first, then the rest
    mean_indices = [i for i, label in enumerate(labels) if label.startswith("Mean")]
    
    if mean_indices:
        mean_index = mean_indices[0]
        mean_handle = handles[mean_index]
        mean_label = labels[mean_index]

        # Remove 'Mean' from the lists
        handles.pop(mean_index)
        labels.pop(mean_index)

        # Combine with Mean at the beginning
        handles = [mean_handle] + handles
        labels = [mean_label] + labels

    # Create legend
    legend = plt.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.3), 
                ncol=legend_ncol, fontsize=8, columnspacing=1.0)
    frame = legend.get_frame()
    frame.set_facecolor('#d9cece')
    
    plt.grid(False)
    
    # Remove top and right spines
    ax = plt.gca()
    ax.set_facecolor('#d9cece')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    

def create_metric_reports(
    results_df: pd.DataFrame, # Changed to accept the full dataframe
    output_path: Path, 
    args: argparse.Namespace, 
    ood: bool,
    return_one_only: bool, 
    correlation_colors: Dict[str, str]
) -> None:
    """
    Creates and saves bar plots and CSV files for both AURC and E-AURC metrics.

    This function acts as an orchestrator, calling a generic worker function
    for each metric that needs to be processed.
    """
    # 1. Determine data modality string for file naming
    if ood and return_one_only:
        data_mod = 'ood'
    elif not ood and return_one_only:
        data_mod = 'id'
    else:
        data_mod = 'id_ood'

    # 2. Generate the report for AURC
    print("-" * 20)
    print("Generating report for AURC...")
    _generate_single_metric_report(
        metric_name='AURC',
        results_df=results_df.sort_values('AURC', ascending=True).reset_index(drop=True), # Sort by AURC for this report
        output_path=output_path,
        args=args,
        data_mod=data_mod,
        correlation_colors=correlation_colors
    )

    # 3. Generate the report for E-AURC
    print("-" * 20)
    print("Generating report for E-AURC...")
    _generate_single_metric_report(
        metric_name='EAURC',
        results_df=results_df.sort_values('EAURC', ascending=True).reset_index(drop=True), # Sort by E-AURC for this report
        output_path=output_path,
        args=args,
        data_mod=data_mod,
        correlation_colors=correlation_colors
    )
    print("-" * 20)
    
def _generate_single_metric_report(metric_name: str, 
    results_df: pd.DataFrame, # Changed to accept the dataframe
    output_path: Path, 
    args: argparse.Namespace,
    data_mod: str, 
    correlation_colors: Dict[str, str]
) -> None:
    """
    Generates and saves a bar plot and CSV file for a single metric.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    metric_name_lower = metric_name.lower()
    
    # The dataframe is already sorted and correct
    print(results_df[['Aggregator', metric_name, f'{metric_name}_std']])

    # Save results to a CSV file
    uq_methods = [uq.strip() for uq in args.uq_methods.split(',')]
    csv_name = f'{args.task}_{args.dataset_name}_{args.variation}_{uq_methods[0]}_{args.decomp}'
    if args.spatial:
        csv_name += f'_{args.spatial}'
    
    csv_dir = output_path.joinpath(f'tables/{metric_name_lower}_{data_mod}')
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_file = csv_dir.joinpath(f'{csv_name}_{metric_name_lower}_{data_mod}_results.csv')

    header = not csv_file.exists() or csv_file.stat().st_size == 0
    results_df.to_csv(csv_file, mode='a' if not header else 'w', index=False, header=header)
    print(f"Data for {metric_name} appended to {csv_file}")
    
    # 3. Create new figure for barplot
    plt.figure(figsize=(6, 5))
    plt.rcParams['axes.grid'] = False
    ax = plt.gca()
    
    # Add title
    plt.suptitle(
        f'iD-OoD failures measured by the AURC w.r.t. model confidence correctness.\n'
        f'Task: {args.task}, Variation: {args.variation}',
        fontsize=16
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    
    # Define method categories for coloring
    method_categories = ["Threshold", "Patch", "Quantile", "Quantile fg."]
    
    # Create method to category mapping
    method_to_category = {}
    for method in results_df['Aggregator']:
        category = "Other"  # Default category
        for cat in method_categories:
            if method.startswith(cat):
                category = cat
                break
        if method.startswith("Mean"):
            category = "Mean"
        elif method.startswith("Equally-w.") or method.startswith("Imbalance-w."):
            category = "Class Average"
        elif method.startswith("GMM"): 
            category = "Spatial"
        method_to_category[method] = category
    
    # Define strategies dictionary for legend
    strategies_dict = {
        "Threshold": ["Threshold 0.3", "Threshold 0.5", "Threshold 0.7"],
        "Patch": ["Patch 10", "Patch 20", "Patch 50"],
        "Quantile": ["Quantile 0.6", "Quantile 0.75", "Quantile 0.9"],
        "Quantile fg.": ["Quantile fg. ratio"],
        "Class Average": ["Equally-w. class avg.", "Imbalance-w. class avg."],
        "Mean": ["Mean"], 
        "Spatial": ["GMM", "GMM_pixel", "GMM_spatial"]
    }
    
    # Define barplot colors (fallback colors)
    barplot_colors = {
        "Threshold": '#1f77b4',
        "Patch": '#ff7f0e', 
        "Quantile": '#2ca02c',
        "Quantile fg.": '#d62728',
        "Class Average": '#9467bd',
        "Mean": '#8c564b',
        "Other": '#e377c2'
    }
    
    # Get colors for each method
    colors = []
    for method in results_df['Aggregator']:
        category = method_to_category[method]
        color = _get_method_color(method, category, correlation_colors, barplot_colors)
        colors.append(color)
    
    # Create bars
    bars = ax.bar(
        results_df['Aggregator'], #range(len(df_sorted)), 
        results_df[metric_name],
        yerr=results_df[f'{metric_name}_std'],
        color=colors,
        capsize=4,
        zorder=3,
        )
    
    # Format bars and axes
    _format_aurc_bars(ax, bars, results_df, show_values=True)
    _format_aurc_axes(ax)
    
    # Add legend
    _add_aurc_legend(ax, strategies_dict, method_to_category, results_df, 
                     correlation_colors, barplot_colors, legend_ncol=3)
    
    # 6. Save the final plot
    _save_metric_barplot(output_path, args, data_mod, metric_name)

def _get_method_color(method_name: str, category: str, correlation_colors: Dict[str, str], 
                     barplot_colors: Dict[str, str]) -> str:
    """Get color for method, prioritizing correlation colors."""
    # Try to get color from correlation colors first
    color = _get_aurc_method_color(method_name, correlation_colors, None)
    if color is not None:
        return color
    
    # Fallback to category-based colors
    return barplot_colors.get(category, '#e377c2')

def _format_aurc_bars(ax, bars, results: pd.DataFrame, show_values: bool = True) -> None:
    """Format bars with values and labels."""
    if show_values:
        # Add AURC values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom'
            )
    
    # Add method labels inside bars
    for bar, label in zip(bars, results['Aggregator']):
        y_offset = 0.005 * 2 * bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_offset,
            label,
            ha="center", va="bottom",
            rotation="vertical",
            fontsize=15,
            zorder=4,
        )

def _format_aurc_axes(ax) -> None:
    """Format axes appearance for AURC barplot."""
    ax.set_ylabel('AURC' + r" $\downarrow$", fontsize=12)  # Lower is better for AURC
    ax.set_ylim(0, 1)
    # Set y-axis limits based on data range
    # y_min = min([bar.get_height() for bar in ax.patches])
    # y_max = max([bar.get_height() for bar in ax.patches])
    # y_range = y_max - y_min
    # ax.set_ylim(max(0, y_min - 0.1 * y_range), y_max + 0.1 * y_range)
    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(axis='y', which='major', labelsize=13)
    ax.set(xticklabels=[])
    ax.tick_params(bottom=False)

def _add_aurc_legend(ax, strategies_dict: Dict, method_to_category: Dict,
                    results: pd.DataFrame, correlation_colors: Dict[str, str], 
                    barplot_colors: Dict[str, str], legend_ncol: int = 3) -> None:
    """Add legend to the AURC barplot."""
    from matplotlib.patches import Patch
    
    legend_elements = []
    for category in strategies_dict.keys():
        # Use the first method's color from this category as representative
        methods_in_category = [method for method in results['Aggregator'] 
                              if method_to_category[method] == category]
        if methods_in_category:
            representative_color = _get_method_color(
                methods_in_category[0], category, correlation_colors, barplot_colors
            )
            legend_elements.append(Patch(facecolor=representative_color, label=category))
    
    # Fallback to original colors if no methods found
    if not legend_elements:
        legend_elements = [
            Patch(facecolor=color, label=category) 
            for category, color in barplot_colors.items()
        ]
    
    ax.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.025),
        fancybox=True,
        shadow=True,
        ncol=legend_ncol
    )

def _save_aurc_plot(output_path: Path, args: argparse.Namespace, ood: bool, return_one_only: bool) -> None:
    """Save the AURC plot to file."""
    if ood and return_one_only:
        data_mod = 'ood'
    elif not ood and return_one_only:
        data_mod = 'id'
    else:
        data_mod = 'id_ood'
    
    uq_methods = [uq.strip() for uq in args.uq_methods.split(',')]
    output_file = output_path.joinpath(
        f'figures/aurc_{data_mod}/{data_mod}_aurc_{args.task}_{args.dataset_name}_{args.variation}_{uq_methods[0]}_{args.decomp}.png'
    )
    
    # Ensure directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    logger.info(f"Plot saved to: {output_file}")
    plt.close()  # Close figure to free memory
    
def _save_metric_barplot(output_path: Path, args: argparse.Namespace, data_mod: str, metric_name: str) -> None:
    """Saves the metric bar plot to the correct directory."""
    metric_name_lower = metric_name.lower()
    figure_dir = output_path.joinpath(f'figures/{metric_name_lower}_{data_mod}')
    figure_dir.mkdir(parents=True, exist_ok=True)

    uq_methods = [uq.strip() for uq in args.uq_methods.split(',')]
    filename = f'{data_mod}_{metric_name_lower}_{args.task}_{args.dataset_name}_{args.variation}_{uq_methods[0]}_{args.decomp}_barplot.png'
    output_file = figure_dir.joinpath(filename)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to prevent label cutoff
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"{metric_name} Barplot saved to: {output_file}")
    plt.close() # Close the figure to free up memory

