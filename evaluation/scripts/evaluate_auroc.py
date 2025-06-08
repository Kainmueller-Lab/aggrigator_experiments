#!/usr/bin/env python

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any, Callable, NamedTuple

from evaluation.data_utils import load_dataset, preload_uncertainty_maps, setup_paths
from evaluation.metrics.auroc_ood import evaluate_all_strategies
from evaluation.visualization.plot_functions import setup_plot_style_auroc, create_auroc_barplot, create_single_auroc_barplot
from evaluation.constants import AUROC_STRATEGIES, NOISE_LEVELS, NOISE_LEVELS_ARCTIQUE, BARPLOTS_COLORS

# ---- Script to evaluate AUROC for OoD detection for various aggregation methods and create comparison plots
    
def clear_csv_file(output_path: Path, task: str, dataset_name: str, variation: str, decomp: str) -> None:
    """Clears the content of the CSV file if it exists."""
    csv_file = output_path.joinpath(f'tables/{task}_{dataset_name}_{variation}_{decomp}_auroc_ood_results.csv')
    # Ensure directory exists
    csv_file.parent.mkdir(exist_ok=True, parents=True)
    if csv_file.exists():
        csv_file.open('w').close()  # Open in write mode to clear contents
        print(f"Cleared content of {csv_file}")
    else:
        print(f"{csv_file} does not exist yet.")

def process_noise_level(dataset: dict, uq_path: Path, metadata_path: Path, gt_list: list, gt_labels: list, task: str, model_noise: int, 
                        variation: str, noise_level: str, output_path: Path, dataset_name: str, decomp: str) -> pd.DataFrame:
    """Process all strategies for a single noise level."""
    print(f"Processing noise level: {noise_level}")
    
    # Preload all uncertainty maps for this noise level
    cached_maps = preload_uncertainty_maps(
        uq_path, metadata_path, gt_list, gt_labels, dataset, task, model_noise, variation, noise_level, dataset_name, decomp
    )
    
    # Evaluate all strategies
    df = evaluate_all_strategies(cached_maps, AUROC_STRATEGIES, noise_level, decomp)
    print(df)
    
    # Save results to CSV
    csv_file = output_path.joinpath(f'tables/{task}_{dataset_name}_{variation}_{decomp}_auroc_ood_results.csv')
    
    # Check if the file exists to handle headers properly
    file_empty = not csv_file.exists() or csv_file.stat().st_size == 0
    
    # Append to CSV (write header only if the file is empty)
    df.to_csv(csv_file, mode='a', index=False, header=file_empty)
    print(f"Data appended to {csv_file}")
    
    return df

def run_auroc_evaluation(args: dict, task: str, variation: str, uq_path: Path, metadata_path: Path, data_path: Path,
                         dataset_name: str, output_path: Path, model_noise: int = 0, decomp: str = "pu") -> None:
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
    dataset_name : str
    output_path : Path
        Path to save output
    model_noise : int, optional
        Model noise level, by default 0
    decomp : str, optional
        Decomposition component, by default "pu"
    """
    # Clear previous results
    clear_csv_file(output_path, task, dataset_name, variation, decomp)
    
    # Define noise levels
    nls = NOISE_LEVELS_ARCTIQUE if dataset_name.startswith('arctique') else NOISE_LEVELS
    
    # Load whole dataset, ground truth masks and AUROC target labels
    dataset, gt_list, gt_labels = load_dataset(
        data_path, 
        '0_00',
        num_workers=2,
        dataset_name=dataset_name,
        task=task
    )
    
    # Process each noise level
    results = []
    for noise_level in nls:
        df = process_noise_level(
            dataset, uq_path, metadata_path, gt_list, gt_labels, task, model_noise, 
            variation, noise_level, output_path, dataset_name, decomp
        )
        results.append(df)
    
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
            output_path
        )
    else:
        create_auroc_barplot(
            results,
            nls,
            BARPLOTS_COLORS,
            AUROC_STRATEGIES,
            task,
            variation,
            dataset_name,
            decomp,
            output_path
        )

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create ranked AUROC plots of unc. heatmaps at image level')
    parser.add_argument('--task', type=str, default='instance', help='Task type (e.g. fgbg, instance, semantic)')
    parser.add_argument('--variation', type=str, default='nuclei_intensity', help='Variation type (e.g. nuclei_intensity, blood_cells, malignancy, texture)')
    parser.add_argument('--uq_path', type=str, default='/home/vanessa/Documents/data/uncertainty_arctique_v1-0-corrected_14/', help='Path to unc. evaluation results')
    # arctique: '/fast/AG_Kainmueller/vguarin/hovernext_trained_models/trained_on_cluster/uncertainty_arctique_v1-0-corrected_14/'
    # lidc: '/fast/AG_Kainmueller/data/ValUES/'
    parser.add_argument('--label_path', type=str, help='Path to labels')
    # arctique: '/fast/AG_Kainmueller/synth_unc_models/data/v1-0-variations/variations/'
    parser.add_argument('--model_noise', type=int, default=0, help='Model noise level')
    parser.add_argument('--decomp', type=str, default='pu', help='Decomposition component (e.g. pu, au, eu)')
    parser.add_argument('--dataset_name', type=str, default='arctique', help='Dataset name (e.g. arctique, lidc)')
    
    return parser.parse_args()

def main():
    # Set up plot style
    setup_plot_style_auroc() 
    
    # Parse arguments 
    args = parse_arguments()
    if args.label_path is None:
        args.label_path = args.uq_path 
    
    #Set paths and make sure output directory exists
    paths = setup_paths(args)
    
    # Run evaluation
    run_auroc_evaluation(
        args=args,
        task=args.task,
        variation=args.variation,
        uq_path=paths.uq_maps,
        metadata_path=paths.metadata,
        data_path=paths.data,
        dataset_name = args.dataset_name, 
        output_path=paths.output,
        model_noise=args.model_noise,
        decomp=args.decomp,
    )

if __name__ == "__main__":
    main()