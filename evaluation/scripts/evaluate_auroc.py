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
from evaluation.constants import (
    AUROC_STRATEGIES, 
    NOISE_LEVELS, 
    NOISE_LEVELS_ARCTIQUE, 
    BARPLOTS_COLORS, 
    AGGREGATOR_NAME_MAPPING,
)

# ---- Script to evaluate AUROC for OoD detection for various aggregation methods and create comparison plots
    
def clear_csv_file(output_path: Path, task: str, dataset_name: str, variation: str, decomp: str, spatial: str = None) -> None:
    """Clears the content of the CSV file if it exists."""
    # Define path and csv name
    csv_name = f'{task}_{dataset_name}_{variation}_{decomp}'
    if spatial: 
        csv_name += f'_{spatial}'
        
    # Define paths for both results and p-values
    auroc_csv_file = output_path.joinpath(f'tables/auroc_gmm/{csv_name}_auroc_ood_results.csv')
    p_value_csv_file = output_path.joinpath(f'tables/auroc_gmm/{csv_name}_p_values.csv')
    
    # Ensure directory exists
    auroc_csv_file.parent.mkdir(exist_ok=True, parents=True)

    # Clear AUROC results file
    if auroc_csv_file.exists():
        auroc_csv_file.open('w').close()
        print(f"Cleared content of {auroc_csv_file}")
    else:
        print(f"{auroc_csv_file} does not exist yet.")

    # Clear p-values file
    if p_value_csv_file.exists():
        p_value_csv_file.open('w').close()
        print(f"Cleared content of {p_value_csv_file}")
    else:
        print(f"{p_value_csv_file} does not exist yet.")
        
    # Clear the single reproducibility file
    repro_csv_dir = output_path.joinpath('tables', 'auroc_reproducibility_repo')
    repro_csv_dir.mkdir(exist_ok=True, parents=True)
    repro_csv_file = repro_csv_dir.joinpath(f'{csv_name}.csv')
    if repro_csv_file.exists(): repro_csv_file.unlink()
    print("Cleared previous CSV files.")

def process_combo_key(concatenated_data: dict, combo_key: str, task: str, variation: str, dataset_name: str, 
                      decomp: str, output_path: Path, n_bootstraps: int, spatial: str = None, ignore_index: int = 0) -> pd.DataFrame:
    """Process all strategies for a single combo key."""
    print(f"Processing combo key: {combo_key}")
    
    # Convert concatenated data to cached maps format
    cached_maps = create_cached_maps_from_concatenated(concatenated_data, combo_key)
    
    # Extract noise level from combo key (e.g., '0_00_0_25' -> '0_25')
    noise_level = combo_key.split('_')[-2] + '_' + combo_key.split('_')[-1]
    
    # Evaluate all strategies to get AUROC stats and p-values
    auroc_df, p_values_df, repro_df = evaluate_all_strategies(
        cached_maps, 
        AUROC_STRATEGIES, 
        noise_level, 
        ignore_index, 
        dataset_name=dataset_name,
        task=task,
        variation=variation,
        decomp=decomp,
        n_bootstraps=n_bootstraps,
        output_path=output_path, 
    )
    print("AUROC Results:\n", auroc_df)
    if not p_values_df.empty:
        print("P-Value Results:\n", p_values_df)
    
    # Save results to CSV
    csv_name = f'{task}_{dataset_name}_{variation}_{decomp}'
    if spatial: 
        csv_name += f'_{spatial}'
    
    auroc_csv_file = output_path.joinpath(f'tables/auroc_gmm/{csv_name}_auroc_ood_results.csv')
    p_value_csv_file = output_path.joinpath(f'tables/auroc_gmm/{csv_name}_p_values.csv')

    # Append AUROC results to CSV
    auroc_file_empty = not auroc_csv_file.exists() or auroc_csv_file.stat().st_size == 0
    auroc_df.to_csv(auroc_csv_file, mode='a', index=False, header=auroc_file_empty)
    print(f"AUROC data appended to {auroc_csv_file}")

    # Append p-value results to a separate CSV
    if not p_values_df.empty:
        p_value_file_empty = not p_value_csv_file.exists() or p_value_csv_file.stat().st_size == 0
        p_values_df.to_csv(p_value_csv_file, mode='a', index=False, header=p_value_file_empty)
        print(f"P-value data appended to {p_value_csv_file}")
    
    if repro_df is not None:
        id_noise_level = combo_key.split('_')[-4] + '_' + combo_key.split('_')[-3]
        ood_noise_level = combo_key.split('_')[-2] + '_' + combo_key.split('_')[-1]
        repro_df['noise_level_id'] = np.where(repro_df['is_ood'] == 0, id_noise_level, ood_noise_level)
        
    return auroc_df, repro_df

def run_auroc_evaluation(concatenated_data: Dict, task: str, variation: str, dataset_name: str, output_path: Path, decomp: str = "pu", 
                         spatial: str = None, noise_levels: List[str] = None, ignore_index: int = 0, n_bootstraps: int = 500) -> None:
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
    ignore_index : int, optional
        Index to ignore in context-aware aggregation strategies
    n_bootstraps : int, optional
        Np. of bootstrap samples to use for cgenertaing confidence intervals
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
    all_repro_dfs = []
    processed_noise_levels = []
    
    for combo_key in combo_keys:
        df, repro_df = process_combo_key(
            concatenated_data, combo_key, task, variation, dataset_name, 
            decomp, output_path, n_bootstraps, spatial, ignore_index
        )
        results.append(df)
        if repro_df is not None: all_repro_dfs.append(repro_df)
    
        # Extract noise level for plotting
        noise_level = combo_key.split('_')[-2] + '_' + combo_key.split('_')[-1]
        processed_noise_levels.append(noise_level)
    
    # Save consolidated reproducibility data
    if all_repro_dfs:
        final_repro_df = pd.concat(all_repro_dfs) #.drop_duplicates(subset=['uq_map_name'], keep='first')
        final_repro_df.drop_duplicates(subset=['uq_map_name', 'noise_level_id'], keep='first', inplace=True)
        final_repro_df.drop(columns=['noise_level_id'], inplace=True)
        
        # Rename columns using the provided mapping
        final_repro_df.rename(columns=AGGREGATOR_NAME_MAPPING, inplace=True)
        final_repro_df.set_index('uq_map_name', inplace=True)
        
        base_name = f'{task}_{dataset_name}_{variation}_{decomp}'
        if spatial: base_name += f'_{spatial}'
        repro_path = output_path.joinpath(f'tables/auroc_reproducibility_repo/{base_name}.csv')
        final_repro_df.to_csv(repro_path)
        print(f"Comprehensive reproducibility data saved to {repro_path}")
    
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
        choices=['fgbg', 'instance', 'semantic', 'crops_vs_weed'], help='Task type'
    )
    parser.add_argument(
        '--variation', type=str, default='nuclei_intensity', 
        choices=['nuclei_intensity', 'blood_cells', 'texture', 'malignancy', 'cityscapes', 'glas_set', 'protists', 'nematodes', 'maize'], help='OoD variation type'
    )
    parser.add_argument(
        '--uq_path', type=str, 
        default='/fast/AG_Kainmueller/vguarin/hovernext_trained_models/trained_on_cluster/uncertainty_arctique_v1-0-corrected_14/', help='Path to unc. evaluation results'
    )
    # arctique: '/fast/AG_Kainmueller/vguarin/hovernext_trained_models/trained_on_cluster/uncertainty_arctique_v1-0-corrected_14/'
    # lidc: '/fast/AG_Kainmueller/data/ValUES/'
    # gta_cityscapes: '/fast/AG_Kainmueller/data/GTA_CityScapes_UQ/'
    # ade20k_cityscapes: '/fast/AG_Kainmueller/data/UQ_maps/ADE20K/'
    # lizard: '/fast/AG_Kainmueller/data/Lizard_AggroUQ/trained_2/'
    # weedsgalore: '/fast/AG_Kainmueller/data/weedsgalore/'
    # wormbodies: '/fast/AG_Kainmueller/data/UQ_maps/wormbodies/'
    parser.add_argument(
        '--label_path', type=str, help='Path to labels'
    )
    # arctique: '/fast/AG_Kainmueller/synth_unc_models/data/v1-0-variations/variations/'
    # gta_cityscapes: '/fast/AG_Kainmueller/data/GTA/'
    # ade20k_cityscapes: '/fast/AG_Kainmueller/data/ADEChallengeData2016/'
    # lizard: '/fast/AG_Kainmueller/data/LizardRaw_new/archive/lizard_tiles.lmdb'
    # weedsgalore: '/fast/AG_Kainmueller/data/weedsgalore/'
    # wormbodies: '/fast/AG_Kainmueller/data/'
    parser.add_argument(
        '--model_noise', type=int, default=0, help='Model noise level'
    )
    parser.add_argument(
        '--decomp', type=str, default='pu', 
        choices=['pu', 'au', 'eu'], help='Information theoretic decomposition component'
    )
    parser.add_argument(
        '--dataset_name', type=str, default='arctique', choices=['arctique', 'lidc', 'lizard', 'gta', 'ade20k', 'wormbodies', 'weedsgalore'], help='Dataset name'
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
    parser.add_argument(
        '--ignore_index', type=int, default=0, 
        help='Background index to ignore in context-aware aggregators'
    )
    parser.add_argument(
        '--model_checkpoint', type=str, default=None,
        help='Pretrained model to pass to extra_info[metadata][model_checkpoint]'
    )
    parser.add_argument(
        '--n_bootstraps', type=int, default=500,
        help='Number of bootstrap samples for AUROC calculation.'
    )
    # ade20k: 'deeplabv3_r50-d8_4xb4-160k_ade20k-512x512'
    return parser.parse_args()

def main():
    # Set up plot style
    setup_plot_style_auroc() 
    
    # Parse arguments 
    args = parse_arguments()
    
    ignore_index = args.ignore_index 
        
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
        'split_path' : None,
        'split' : None,
        'model_checkpoint' : args.model_checkpoint, 
        'real_task' : args.task
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
        noise_levels=noise_levels,
        ignore_index=ignore_index,
        n_bootstraps=args.n_bootstraps
    )

if __name__ == "__main__":
    main()