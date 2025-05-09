#!/usr/bin/env python

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any, Callable, NamedTuple

from evaluation.data_utils import load_dataset, preload_uncertainty_maps, setup_paths
from evaluation.metrics.auroc_ood_detection import evaluate_all_strategies
from evaluation.visualization.plot_functions import setup_plot_style, create_auroc_barplot
from evaluation.constants import AUROC_STRATEGIES, NOISE_LEVELS, BARPLOTS_COLORS

# ---- Script to evaluate AUROC for OoD detection for various aggregation methods and create comparison plots
    
def clear_csv_file(output_path: Path, task: str) -> None:
    """Clears the content of the CSV file if it exists."""
    csv_file = output_path.joinpath(f'tables/{task}_auroc_ood_detect_results.csv')
    # Ensure directory exists
    csv_file.parent.mkdir(exist_ok=True, parents=True)
    if csv_file.exists():
        csv_file.open('w').close()  # Open in write mode to clear contents
        print(f"Cleared content of {csv_file}")
    else:
        print(f"{csv_file} does not exist yet.")

def process_noise_level(uq_path: Path, metadata_path: Path, gt_list: list, task: str, model_noise: int, 
                        variation: str, noise_level: str, output_path: Path) -> pd.DataFrame:
    """Process all strategies for a single noise level."""
    print(f"Processing noise level: {noise_level}")
    
    # Preload all uncertainty maps for this noise level
    cached_maps = preload_uncertainty_maps(
        uq_path, metadata_path, gt_list, task, model_noise, variation, noise_level
    )
    
    # Evaluate all strategies
    df = evaluate_all_strategies(cached_maps, AUROC_STRATEGIES, noise_level)
    print(df)
    
    # Save results to CSV
    csv_file = output_path.joinpath(f'tables/{task}_auroc_ood_detect_results.csv')
    
    # Check if the file exists to handle headers properly
    file_empty = not csv_file.exists() or csv_file.stat().st_size == 0
    
    # Append to CSV (write header only if the file is empty)
    df.to_csv(csv_file, mode='a', index=False, header=file_empty)
    print(f"Data appended to {csv_file}")
    
    return df

def run_auroc_evaluation(task: str, variation: str, uq_path: Path, metadata_path: Path, data_path: Path,
                         output_path: Path, model_noise: int = 0, decomp: str = "pu") -> None:
    """
    Create comparative bar plots of image-level AUROC values for different noise levels and UQ methods.
    
    Parameters
    ----------
    task : str
        Task type ('instance' or 'semantic')
    variation : str
        Variation type
    uq_path : Path
        Path to uncertainty maps
    metadata_path : Path
        Path to metadata files
    data_path : Path
        Path to dataset
    output_path : Path
        Path to save output
    model_noise : int, optional
        Model noise level, by default 0
    decomp : str, optional
        Decomposition component, by default "pu"
    """
    # Clear previous results
    clear_csv_file(output_path, task)
    
    # Load zero-risk labels
    _, gt_list = load_dataset(
        data_path, 
        '0_00',
        is_ood='ood',
        num_workers=2,
        dataset_name='arctique'
    )
    
    # Process each noise level
    results = []
    for noise_level in NOISE_LEVELS:
        df = process_noise_level(
            uq_path, metadata_path, gt_list, task, 
            model_noise, variation, noise_level, output_path
        )
        results.append(df)
    
    # Create plots
    create_auroc_barplot(
        results,
        NOISE_LEVELS,
        BARPLOTS_COLORS,
        AUROC_STRATEGIES,
        task,
        variation,
        output_path
    )

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create ranked AUROC plots of unc. heatmaps at image level')
    parser.add_argument('--task', type=str, default='instance', help='Task type (e.g., instance or semantic)')
    parser.add_argument('--variation', type=str, default='nuclei_intensity', help='Variation type (e.g., nuclei_intensity or blood_cells)')
    parser.add_argument('--uq_path', type=str, default='/home/vanessa/Documents/data/uncertainty_arctique_v1-0-corrected_14/', help='Path to unc. evaluation results')
    parser.add_argument('--label_path', type=str, default='/home/vanessa/Desktop/synth_unc_models/data/v1-0-variations/variations/', help='Path to labels')
    parser.add_argument('--model_noise', type=int, default=0, help='Model noise level')
    parser.add_argument('--decomp', type=str, default='pu', help='Decomposition component (e.g. pu, au, eu)')
    
    return parser.parse_args()

def main():
    # Set up plot style
    setup_plot_style() 
    
    # Parse arguments 
    args = parse_arguments()
    
    #Set paths and make sure output directory exists
    paths = setup_paths(args)
    
    # Run evaluation
    run_auroc_evaluation(
        task=args.task,
        variation=args.variation,
        uq_path=paths.uq_maps,
        metadata_path=paths.metadata,
        data_path=paths.data,
        output_path=paths.output,
        model_noise=args.model_noise,
        decomp=args.decomp
    )

if __name__ == "__main__":
    main()