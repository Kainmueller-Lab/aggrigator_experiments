import sys
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any, Tuple, Dict, NamedTuple, Callable

from evaluation.constants import (CLASS_NAMES_ARCTIQUE, 
                       CLASS_NAMES_LIZARD, 
                       AUROC_STRATEGIES, 
                       BACKGROUND_FREE_STRATEGIES, 
                       COLORS)
from evaluation.metrics.accuracy_metrics import per_tile_metrics
from evaluation.metrics.selective_risk_coverage import compute_selective_risks_coverage
from evaluation.data_utils import (DataPaths,
                                   AnalysisResults,
                                   setup_paths, 
                                   load_predictions, 
                                   load_dataset, 
                                   load_unc_maps, 
                                   validate_indices,
                                   select_strategies,
                                   rescale_maps,
                                   _process_gt_masks,
                                   remove_background_only_images)
from evaluation.visualization.plot_functions import setup_plot_style_aurc, create_selective_risks_coverage_plot
    
# ---- Configuration Functions ----

def variation_name():
    # When there is no clear id and ood distinction in the inputs 
    return { 
        'lizard' : 'LizardData'
    } 

def clear_csv_file(output_path: Path, args: argparse.Namespace) -> None:
    """Clears the content of the CSV file if it exists."""
    csv_file = output_path.joinpath(
        f'tables/aurc_data_{args.aggregator_type}_aggr_multi_uq_methods_{args.task}_{args.variation}_{args.data_mod}.csv'
    )
    # Ensure directory exists
    csv_file.parent.mkdir(exist_ok=True, parents=True)
    
    if csv_file.exists():
        csv_file.open('w').close()  # Open in write mode to clear contents
        print(f"Cleared content of {csv_file}")
    else:
        print(f"{csv_file} does not exist yet.")

def parse_args():
    parser = argparse.ArgumentParser(description='Create accuracy-rejection curves for aggregators')
    parser.add_argument('--task', type=str, default='semantic', help='Task type (e.g. fgbg, instance, semantic)')
    parser.add_argument('--variation', type=str, help='Variation type (e.g. nuclei_intensity, blood_cells, malignancy, texture)')
    parser.add_argument('--uq_path', type=str, default='/home/vanessa/Documents/data/uncertainty_arctique_v1-0-corrected_14/', help='Path to unc. evaluation results')
    # arctique: '/fast/AG_Kainmueller/vguarin/hovernext_trained_models/trained_on_cluster/uncertainty_arctique_v1-0-corrected_14/'
    # lizard:  '/fast/AG_Kainmueller/vguarin/hovernext_trained_models/trained_on_cluster/uncertainty_lizard_convnextv2_tiny_3' 
    # lidc: '/fast/AG_Kainmueller/data/ValUES/'
    parser.add_argument('--label_path', type=str, help='Path to labels')
    # arctique: '/fast/AG_Kainmueller/synth_unc_models/data/v1-0-variations/variations/'
    # lizard:  '/fast/AG_Kainmueller/vguarin/synthetic_uncertainty/data/LizardData/' 
    parser.add_argument('--model_noise', type=int, default=0, help='Mask noise level with which the model was trained')
    parser.add_argument('--image_noise', type=str, default='0_00', help='Image noise level on which the model is evaluated')
    parser.add_argument('--uq_methods', type=str, default='tta,softmax,ensemble,dropout', help='Comma-separated list of UQ methods to evaluate')
    parser.add_argument('--decomp', type=str, default='pu', help='Decomposition component (e.g. pu, au, eu)')
    parser.add_argument('--data_mod', type=str, default='id', help='Data Modality (e.g. ood or id)')
    parser.add_argument('--aggregator_type', type=str, default='non-pi', help='Aggregator Property (e.g. proportion-invariant or non-pi)' )
    parser.add_argument('--num_workers', type=int, default=4, help='No. of workers for parallel processing' )
    parser.add_argument('--dataset_name', type=str, default='arctique', help='Selected dataset (e.g. arctique, lizard, lidc)' )
    
    return parser.parse_args()

# ---- Analysis Functions ----

def run_aurc_evaluation(args: argparse.Namespace, paths: DataPaths) -> None:
    """
    Run the AURC evaluation pipeline.
    
    Args:
        args: Command line arguments
        output_path: Path to save output
    """
    
    # Extract parameters from arguments
    task = args.task
    model_noise = args.model_noise
    image_noise = args.image_noise
    decomp = args.decomp
    aggregator_type = args.aggregator_type
    num_workers = args.num_workers
    dataset_name = args.dataset_name
    variation = args.variation #if args.variation else 'LizardData'
    uq_methods = args.uq_methods.split(',')
    data_mod = args.data_mod
    
    # Select appropriate strategies based on aggregator type and extract method names for plotting
    strategies, method_names = select_strategies(aggregator_type)
    
    # Load dataset and ground truth
    dataset, gt_list_old = load_dataset(
        data_path=paths.data, 
        image_noise=image_noise,
        num_workers=num_workers,
        dataset_name=dataset_name,
        return_id_only=(data_mod == 'id')
    )
        
    # Store results for all methods
    all_results = {
        "aurc": [],
        "coverages": None,
        "selective_risks": []
    }
    # Process each UQ method
    for idx, uq_method in enumerate(uq_methods):
        print(f"\n=== Processing UQ method: {uq_method} ===")
        
        # Load prediction data for current UQ method
        pred_list= load_predictions(
            paths,
            model_noise,
            variation,
            image_noise,
            uq_method,
            dataset_name,
            # (dataset_name== 'arctique' or dataset_name== 'lizard'),
        )
                
        # Load and check metadata indices for consistency
        gt_list = validate_indices(
            args, paths.metadata, uq_method, dataset, gt_list_old, dataset_name
        )

        # Analyze uncertainty and generate results
        print(f"Analyzing uncertainty using {aggregator_type} aggregation strategies with {uq_method}")
        results = compute_selective_risks_coverage(
            gt_list,
            pred_list,
            paths,
            task,
            model_noise,
            uq_method,
            decomp,
            variation,
            image_noise,
            strategies,
            num_workers,
            dataset_name
        )
        
        # Store results
        all_results["coverages"] = results["coverages"]
        all_results["selective_risks"].append(results["selective_risks"])
        all_results["aurc"].append(results["aurc"])

    # Calculate mean and std across all UQ methods
    mean_aurc = np.mean(np.array(all_results["aurc"]), axis=0)
    mean_selective_risks = np.mean(np.array(all_results["selective_risks"]), axis=0)
    std_selective_risks = np.std(np.array(all_results["selective_risks"]), axis=0)
    
    # Create final results structure for plotting
    final_results = AnalysisResults(
        mean_aurc=mean_aurc,
        coverages=all_results["coverages"],
        mean_selective_risks=mean_selective_risks,
        std_selective_risks=std_selective_risks
    )
    
    # Create plot
    create_selective_risks_coverage_plot(method_names, final_results, paths.output, args)

def main():
    # Set up plot style
    setup_plot_style_aurc()
    
    # Parse arguments 
    args = parse_args()
    if args.label_path is None:
        args.label_path = args.uq_path 
    if not args.variation:
        alt_names = variation_name()
        args.variation = alt_names[args.dataset_name]
    
    #Set paths and make sure output directory exists
    paths = setup_paths(args)
    
    # Clean Excel file for plot
    clear_csv_file(paths.output, args)
    
    # Run evaluation
    run_aurc_evaluation(args, paths)

if __name__ == "__main__":
    main()