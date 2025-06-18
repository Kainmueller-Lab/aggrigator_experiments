#!/usr/bin/env python

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

from evaluation.data_utils import (
    load_dataset, 
    preload_uncertainty_maps, 
    setup_paths, 
    load_dataset_abstract_class, 
    generate_combo_keys, 
    create_cached_maps_from_concatenated
)
from evaluation.metrics.auroc_ood import evaluate_all_strategies
from evaluation.visualization.plot_functions import setup_plot_style_auroc, create_auroc_barplot, create_single_auroc_barplot
from evaluation.constants import AUROC_STRATEGIES, NOISE_LEVELS, NOISE_LEVELS_ARCTIQUE, BARPLOTS_COLORS

# ---- Script to evaluate AUROC for OoD detection for various aggregation methods and create comparison plots
    
def clear_csv_file(output_path: Path, task: str, dataset_name: str, variation: str, decomp: str, spatial: str = None) -> None:
    """Clears the content of the CSV file if it exists."""
    # Define path and csv name
    csv_name = f'{task}_{dataset_name}_{variation}_{decomp}'
    if spatial: 
        csv_name += f'_{spatial}'
    csv_file = output_path.joinpath(f'tables/{csv_name}_auroc_ood_results.csv')
    
    # Ensure directory exists
    csv_file.parent.mkdir(exist_ok=True, parents=True)
    if csv_file.exists():
        csv_file.open('w').close()  # Open in write mode to clear contents
        print(f"Cleared content of {csv_file}")
    else:
        print(f"{csv_file} does not exist yet.")

def process_combo_key(concatenated_data: dict, combo_key: str, task: str, variation: str, 
                     dataset_name: str, decomp: str, output_path: Path, spatial: str = None) -> pd.DataFrame:
    """Process all strategies for a single combo key."""
    print(f"Processing combo key: {combo_key}")
    
    # Convert concatenated data to cached maps format
    cached_maps = create_cached_maps_from_concatenated(concatenated_data, combo_key)
    
    # Extract noise level from combo key (e.g., '0_00_0_25' -> '0_25')
    noise_level = combo_key.split('_')[-2] + '_' + combo_key.split('_')[-1]
    
    # Evaluate all strategies
    df = evaluate_all_strategies(cached_maps, AUROC_STRATEGIES, noise_level, decomp)
    print(df)
    
    # Save results to CSV
    csv_name = f'{task}_{dataset_name}_{variation}_{decomp}'
    if spatial: 
        csv_name += f'_{spatial}'
    csv_file = output_path.joinpath(f'tables/{csv_name}_auroc_ood_results.csv')
    
    # Check if the file exists to handle headers properly
    file_empty = not csv_file.exists() or csv_file.stat().st_size == 0
    
    # Append to CSV (write header only if the file is empty)
    df.to_csv(csv_file, mode='a', index=False, header=file_empty)
    print(f"Data appended to {csv_file}")
    
    return df

def run_auroc_evaluation(concatenated_data: Dict, task: str, variation: str, dataset_name: str, output_path: Path, 
                         decomp: str = "pu", spatial: str = None, noise_levels: List[str] = None) -> None:
    """
    Create comparative bar plots of image-level AUROC values for different combo keys and UQ methods.
    
    Parameters
    ----------
    concatenated_data : Dict
        The concatenated data from load_dataset_abstract_class
    task : str
        Task type ('instance' or 'semantic')
    variation : str
        Variation type
    dataset_name : str
    output_path : Path
        Path to save output
    decomp : str, optional
        Decomposition component, by default "pu"
    spatial : str, optional
        Spatial measure to weigh the uncertainty maps, by default None
    noise_levels : List[str], optional
        List of noise levels to generate combo keys
    """
    # Clear previous results
    clear_csv_file(output_path, task, dataset_name, variation, decomp, spatial)
    
    # Generate combo keys from noise levels or extract from concatenated_data
    if noise_levels:
        combo_keys = generate_combo_keys(noise_levels)
    else:
        # Extract combo keys from concatenated_data
        # Assuming all uq_methods have the same combo keys
        first_uq_method = next(iter(concatenated_data.keys()))
        combo_keys = list(concatenated_data[first_uq_method].keys())
    
    print(f"Processing combo keys: {combo_keys}")
    
    # Process each combo key
    results = []
    processed_noise_levels = []
    
    for combo_key in combo_keys:
        df = process_combo_key(
            concatenated_data, combo_key, task, variation, 
            dataset_name, decomp, output_path, spatial
        )
        results.append(df)
        
        # Extract noise level for plotting
        noise_level = combo_key.split('_')[-2] + '_' + combo_key.split('_')[-1]
        processed_noise_levels.append(noise_level)
    
    # Create plots
    if len(results) == 1:
        create_single_auroc_barplot(
            results[0],
            BARPLOTS_COLORS,
            AUROC_STRATEGIES,
            task,
            variation,
            dataset_name,
            decomp,
            output_path,
            spatial
        )
    else:
        create_auroc_barplot(
            results,
            processed_noise_levels,
            BARPLOTS_COLORS,
            AUROC_STRATEGIES,
            task,
            variation,
            dataset_name,
            decomp,
            output_path,
            spatial
        )

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create ranked AUROC plots of unc. heatmaps at image level')
    parser.add_argument(
        '--task', type=str, default='instance', 
        choices=['fgbg', 'instance', 'semantic'], help='Task type'
    )
    parser.add_argument(
        '--variation', type=str, default='nuclei_intensity', 
        choices=['nuclei_intensity', 'blood_cells', 'texture', 'malignancy'], help='OoD variation type'
    )
    parser.add_argument(
        '--uq_path', type=str, 
        default='/home/vanessa/Documents/data/uncertainty_arctique_v1-0-corrected_14/', help='Path to unc. evaluation results'
    )
    # arctique: '/fast/AG_Kainmueller/vguarin/hovernext_trained_models/trained_on_cluster/uncertainty_arctique_v1-0-corrected_14/'
    # lidc: '/fast/AG_Kainmueller/data/ValUES/'
    parser.add_argument(
        '--label_path', type=str, help='Path to labels'
    )
    # arctique: '/fast/AG_Kainmueller/synth_unc_models/data/v1-0-variations/variations/'
    parser.add_argument(
        '--model_noise', type=int, default=0, help='Model noise level'
    )
    parser.add_argument(
        '--decomp', type=str, default='pu', 
        choices=['pu', 'au', 'eu'], help='Information theoretic decomposition component'
    )
    parser.add_argument(
        '--dataset_name', type=str, default='arctique', choices=['arctique', 'lidc', 'lizard'], help='Dataset name'
    )
    parser.add_argument(
        '--spatial', type=str, choices=['high_eds', 'low_eds', 'high_moran', 'low_moran'], 
        help='if not none indicate which type of spatially weighted uncertainty maps to use'
    )
    parser.add_argument(
        '--image_noise', type=str, default='0_00,0_25,0_50,0_75,1_00', 
        help='Comma-separated list of image noise levels'
    )
    parser.add_argument(
        '--uq_methods', type=str, default='softmax,ensemble,dropout,tta', 
        help='Comma-separated list of image noise levels'
    )
    parser.add_argument(
        '--metadata', type=str, default=True, 
        help='Read the metadata file if it is stored in the old UQ_metadata format'
    )
    return parser.parse_args()

def main():
    # Set up plot style
    setup_plot_style_auroc() 
    
    # Parse arguments 
    args = parse_arguments()
        
    if args.label_path is None:
        args.label_path = args.uq_path 
        
    if args.spatial and args.decomp != 'pu':
        raise ValueError('Spatially weighted uncertainty maps calculated only for total predictive uncertainty')
    
    # define parameters along which to loop
    noise_levels = [noise.strip() for noise in args.image_noise.split(',')]
    uq_methods = [uq.strip() for uq in args.uq_methods.split(',')]
    
    if 'softmax' in args.uq_methods and args.decomp != 'pu':
        raise ValueError('Softmax uncertainty maps cannot be decomposed')
    
    # Define **kwargs dictionary for dataloaders
    extra_info = {
        'task' : args.task,
        'variation' : args.variation,
        'model_noise' : args.model_noise,
        'decomp' : args.decomp,
        'spatial' : args.spatial,
        'metadata' : args.metadata,
    }
        
    # Set paths and make sure output directory exists
    paths = setup_paths(args)
    
    # Load whole input, ground truth masks, uq maps, predictions, and AUROC target labels
    concatenated_data = load_dataset_abstract_class(
        paths=paths, 
        image_noises=noise_levels,
        num_workers=2,
        extra_info=extra_info,
        dataset_name=args.dataset_name,
        task=args.task,
        uq_methods=uq_methods
    )
    
    # Run evaluation with new function
    run_auroc_evaluation(
        concatenated_data=concatenated_data,
        task=args.task,
        variation=args.variation,
        dataset_name=args.dataset_name,
        output_path=paths.output,
        decomp=args.decomp,
        spatial=args.spatial,
        noise_levels=noise_levels
    )

if __name__ == "__main__":
    main()