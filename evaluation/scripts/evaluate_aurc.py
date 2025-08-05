import sys
import argparse
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any, Tuple, Dict, NamedTuple, Callable

from evaluation.constants import (
    CLASS_NAMES_ARCTIQUE, 
    CLASS_NAMES_LIZARD, 
    AUROC_STRATEGIES, 
    BACKGROUND_FREE_STRATEGIES, 
    COLORS,
    AGGREGATOR_NAME_MAPPING
)
from evaluation.metrics.selective_risk_coverage import (
    compute_uncertainty_and_accuracy_scores,
    _compute_bootstrapped_aurc_stats,
    _perform_pairwise_wilcoxon_tests_on_eaurc,
    # compute_selective_risks_coverage
)
from evaluation.data_utils import (DataPaths,
                                   AnalysisResults,
                                   setup_paths, 
                                   select_strategies,
                                   load_dataset_abstract_class, 
                                   create_cached_maps_from_concatenated)
from evaluation.visualization.plot_functions import setup_plot_style_aurc, create_selective_risks_coverage_plot
    
# ---- Configuration Functions ----

def variation_name():
    # When there is no clear id and ood distinction in the inputs 
    return { 
        'lizard' : 'LizardData'
    } 

def clear_csv_file(output_path: Path, args: argparse.Namespace) -> None:
    """Clears the content of the CSV file if it exists."""
    # Check selective risk-classification file's existence 
    csv_file = output_path.joinpath(
        f'tables/aurc_{args.data_mod}/aurc_data_{args.aggregator_type}_aggr_multi_uq_methods_{args.real_task}_{args.variation}_{args.data_mod}.csv'
    )
    # Ensure directory exists
    csv_file.parent.mkdir(exist_ok=True, parents=True)
    
    if csv_file.exists():
        csv_file.open('w').close()  # Open in write mode to clear contents
        print(f"Cleared content of {csv_file}")
    else:
        print(f"{csv_file} does not exist yet.")
        
    # Check aurc barplots file's existence
    aurc_csv_name = f'{args.real_task}_{args.dataset_name}_{args.variation}_{args.decomp}'
    if args.spatial: 
        aurc_csv_name += f'_{args.spatial}'
    aurc_csv_file = output_path.joinpath(f'tables/aurc_{args.data_mod}/{aurc_csv_name}_aurc_{args.data_mod}_results.csv')
    
    if aurc_csv_file.exists():
        aurc_csv_file.open('w').close()  # Open in write mode to clear contents
        print(f"Cleared content of {aurc_csv_file}")
    else:
        print(f"{aurc_csv_file} does not exist yet.")
    
    # Check aurc barplots file's existence
    eaurc_csv_name = f'{args.real_task}_{args.dataset_name}_{args.variation}_{args.decomp}'
    if args.spatial: 
        eaurc_csv_name += f'_{args.spatial}'
    eaurc_csv_file = output_path.joinpath(f'tables/eaurc_{args.data_mod}/{aurc_csv_name}_eaurc_{args.data_mod}_results.csv')
    
    if eaurc_csv_file.exists():
        eaurc_csv_file.open('w').close()  # Open in write mode to clear contents
        print(f"Cleared content of {eaurc_csv_file}")
    else:
        print(f"{eaurc_csv_file} does not exist yet.")
        
    # --- Clear the reproducibility file ---
    repro_csv_dir = output_path.joinpath('tables', 'eaurc_reproducibility_repo')
    repro_csv_dir.mkdir(exist_ok=True, parents=True)
    repro_csv_file = repro_csv_dir.joinpath(f'{eaurc_csv_name}.csv')
    if repro_csv_file.exists():
        repro_csv_file.unlink()
        print(f"Cleared {repro_csv_file}")

def parse_args():
    parser = argparse.ArgumentParser(description='Create accuracy-rejection curves for aggregators')
    parser.add_argument(
        '--task', type=str, default='instance', 
        choices=['fgbg', 'instance', 'semantic', 'panoptic', 'crops_vs_weed'], help='Task type'
    )
    parser.add_argument(
        '--variation', type=str, 
        choices=['nuclei_intensity', 'blood_cells', 'texture', 'malignancy', 'cityscapes', 'protists', 'nematodes', 'glas_set_sem', 'glas_set_inst', 'maize'], help='OoD variation type'
    )
    parser.add_argument(
        '--uq_path', type=str, 
        default='/home/vanessa/Documents/data/uncertainty_arctique_v1-0-corrected_14/', help='Path to unc. evaluation results'
    )
    # arctique: '/fast/AG_Kainmueller/vguarin/hovernext_trained_models/trained_on_cluster/uncertainty_arctique_v1-0-corrected_14/'
    # lizard:  '/fast/AG_Kainmueller/data/Lizard_AggroUQ/trained_2/'; old_one: '/fast/AG_Kainmueller/vguarin/hovernext_trained_models/trained_on_cluster/uncertainty_lizard_convnextv2_tiny_3' 
    # lidc: '/fast/AG_Kainmueller/data/ValUES/'
    # gta_cityscapes: '/fast/AG_Kainmueller/data/GTA_CityScapes_UQ/'
    # ade20k_cityscapes: '/fast/AG_Kainmueller/data/UQ_maps/ADE20K/'
    # weedsgalore: '/fast/AG_Kainmueller/data/UQ_maps/weedsgalore/'
    # wormbodies: '/fast/AG_Kainmueller/data/UQ_maps/wormbodies/'
    parser.add_argument(
        '--label_path', type=str, help='Path to labels'
    )
    # arctique: '/fast/AG_Kainmueller/synth_unc_models/data/v1-0-variations/variations/'
    # lizard: '/fast/AG_Kainmueller/data/LizardRaw_new/archive/lizard_tiles.lmdb'; old_one: '/fast/AG_Kainmueller/vguarin/synthetic_uncertainty/data/LizardData/' 
    # gta_cityscapes: '/fast/AG_Kainmueller/data/GTA/'
    # ade20k_cityscapes: '/fast/AG_Kainmueller/data/ADEChallengeData2016/'
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
        '--dataset_name', type=str, default='arctique', 
        choices=['arctique', 'lidc', 'lizard', 'gta', 'ade20k', 'weedsgalore', 'wormbodies'], help='Dataset name'
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
    # ade20k: 'deeplabv3_r50-d8_4xb4-160k_ade20k-512x512'
    parser.add_argument(
        '--data_mod', type=str, default='id', 
        help='Data Modality (e.g. ood, id or id_ood)'
    )
    parser.add_argument(
        '--aggregator_type', type=str, default='non-pi', 
        help='Aggregator Property (e.g. proportion-invariant or non-pi)'
    )
    parser.add_argument('--num_workers', type=int, default=1, help='No. of workers for parallel processing' )
    parser.add_argument('--n_bootstraps', type=int, default=500, help='Number of bootstrap samples for AURC/E-AURC calculation.')
    
    return parser.parse_args()

# ---- Analysis Functions ----

def run_aurc_evaluation(args: argparse.Namespace, paths: DataPaths) -> None:
    """Run the AURC evaluation pipeline with bootstrapping and statistical tests."""
    ood = (args.data_mod == 'ood')
    return_one_only = args.data_mod != 'id_ood'
    noise_levels = [noise.strip() for noise in args.image_noise.split(',')]
    uq_methods = [uq.strip() for uq in args.uq_methods.split(',')]
    
    strategies, method_names = select_strategies(args.aggregator_type)
    # Ensure GMM placeholders are always included for processing
    strategies.setdefault('Spatial', {})['GMM'] = (None, None)
    strategies.setdefault('Spatial', {})['GMM_pixel'] = (None, None)
    strategies.setdefault('Spatial', {})['GMM_spatial'] = (None, None)
    method_names_ordered = [name for cat in strategies.values() for name in cat.keys()]

    extra_info = {
        'task': args.task, 'variation': args.variation, 'model_noise': args.model_noise,
        'decomp': args.decomp, 'spatial': args.spatial, 'metadata': args.metadata,
        'split_path': None, 'split': None, 'model_checkpoint': args.model_checkpoint,
        'real_task': args.real_task
    }

    concatenated_data = load_dataset_abstract_class(
        paths=paths, image_noises=noise_levels, num_workers=1, extra_info=extra_info,
        dataset_name=args.dataset_name, task=args.task, return_one_only=return_one_only,
        uq_methods=uq_methods, ood=ood
    )

    first_uq_method = next(iter(concatenated_data.keys()))
    combo_keys = list(concatenated_data[first_uq_method].keys())
    print(f"Processing combo keys: {combo_keys}")

    all_repro_dfs = [] 

    for combo_key in combo_keys:
        print(f"\n--- Processing Combo Key: {combo_key} ---")
        cached_maps = create_cached_maps_from_concatenated(concatenated_data, combo_key, args, args.real_task)
        
        all_uq_unc_scores, all_uq_acc_scores = [], []
        # Temporary list for chunks from the CURRENT combo_key
        combo_repro_chunks = []

        for uq_method in list(cached_maps.keys()):
            print(f"\n=== Collecting scores for UQ method: {uq_method} ===")
            
            unc_scores, acc_scores, repro_df_chunk = compute_uncertainty_and_accuracy_scores(
                uq_maps=cached_maps[uq_method]['maps'],
                gt_list=cached_maps[uq_method]['real_masks'],
                pred_list=cached_maps[uq_method]['masks'],
                sample_names=cached_maps[uq_method]['sample_names'],
                gt_labels=cached_maps[uq_method]['gt_labels'],
                paths=paths, task=args.task, model_noise=args.model_noise, uq_method=uq_method,
                decomp=args.decomp, variation=args.variation, data_noise=noise_levels,
                strategies=strategies, num_workers=args.num_workers, dataset_name=args.dataset_name,
                ood=ood, return_one_only=return_one_only
            )
            all_uq_unc_scores.append(unc_scores)
            all_uq_acc_scores.append(acc_scores)
            if repro_df_chunk is not None:
                combo_repro_chunks.append(repro_df_chunk)
        
        # --- RESTORED LOGIC ---
        # Process the reproducibility chunks for the current combo_key
        if combo_repro_chunks:
            # Set the multi-index on each chunk for correct averaging
            for i in range(len(combo_repro_chunks)):
                combo_repro_chunks[i].set_index(['uq_map_name', 'is_ood'], inplace=True)

            # Concatenate and average over UQ methods, keeping the multi-index
            averaged_chunk = pd.concat(combo_repro_chunks).groupby(level=[0, 1]).mean().reset_index()
            
            # Create the crucial temporary noise_level_id for de-duplication
            if len(combo_key.split('_')) > 2: # Handles 'id_vs_ood' keys like '0_00_1_00'
                id_noise = combo_key.split('_')[0] + '_' + combo_key.split('_')[1]
                ood_noise = combo_key.split('_')[2] + '_' + combo_key.split('_')[3]
            else: # Handles single noise keys like '0_00'
                id_noise = combo_key
                ood_noise = combo_key
            averaged_chunk['noise_level_id'] = np.where(averaged_chunk['is_ood'] == 0, id_noise, ood_noise)
            
            # Add the fully processed chunk to our master list
            all_repro_dfs.append(averaged_chunk)

        # Step 2: Average the raw scores across all UQ methods
        final_unc_scores = np.mean(np.array(all_uq_unc_scores), axis=0)
        final_acc_scores = np.mean(np.array(all_uq_acc_scores), axis=0) # Accuracy is the same, but this keeps shape consistent

        # Step 3: Perform bootstrapping on the final, averaged scores
        bootstrapped_stats, eaurc_bootstrap_samples = _compute_bootstrapped_aurc_stats(
            final_unc_scores, final_acc_scores, method_names_ordered, args.n_bootstraps
        )

        # Step 4: Perform pairwise Wilcoxon tests on the E-AURC bootstrap samples
        p_values_df = _perform_pairwise_wilcoxon_tests_on_eaurc(eaurc_bootstrap_samples)
        
        # Step 5: Save results to CSV
        base_name = f'{args.real_task}_{args.dataset_name}_{args.variation}_{args.decomp}'
        if args.spatial: base_name += f'_{args.spatial}'
        
        # Save main AURC/E-AURC results
        results_df = pd.DataFrame({
            'Aggregator': method_names_ordered,
            'AURC': bootstrapped_stats['mean_aurc'],
            'AURC_std': bootstrapped_stats['std_aurc'],
            'EAURC': bootstrapped_stats['mean_eaurc'],
            'EAURC_std': bootstrapped_stats['std_eaurc']
        })
        
        # Sort the dataframe for display and plotting. This is the only place sorting happens.
        results_df_sorted = results_df.sort_values('EAURC').reset_index(drop=True)
        print("\n--- Sorted E-AURC Results ---")
        print(results_df_sorted)
        
        # results_path = paths.output.joinpath(f'tables/aurc_{args.data_mod}/{base_name}_aurc_{args.data_mod}_results.csv')
        # results_df.to_csv(results_path, index=False)
        # print(f"\nAURC/E-AURC results saved to {results_path}")
        # print(results_df)

        # Save p-value results
        p_values_path = paths.output.joinpath(f'tables/eaurc_{args.data_mod}/{base_name}_eaurc_{args.data_mod}_p_values.csv')
        p_values_df.to_csv(p_values_path, index=False)
        print(f"E-AURC p-value results saved to {p_values_path}")
        print(p_values_df)
        
        # Step 7: Create plot with confidence intervals
        final_results = AnalysisResults(
            mean_aurc=np.array(results_df_sorted['AURC']),
            std_aurc=np.array(results_df_sorted['AURC_std']),
            mean_eaurc=np.array(results_df_sorted['EAURC']),
            std_eaurc=np.array(results_df_sorted['EAURC_std']),
            coverages=np.array(bootstrapped_stats['coverages'])[0],
            # We need to reorder the selective risks to match the sorted dataframe
            mean_selective_risks=np.array(bootstrapped_stats['mean_selective_risks'])[results_df_sorted.index].T,
            std_selective_risks=np.array(bootstrapped_stats['std_selective_risks'])[results_df_sorted.index].T
        )
        # Pass the sorted method names to the plotting function
        create_selective_risks_coverage_plot(
            results_df_sorted['Aggregator'].tolist(), 
            final_results, paths.output, args, ood
        )
        
        # --- FINAL REPRODUCIBILITY DATA HANDLING (AFTER ALL COMBO KEYS ARE PROCESSED) ---
        if all_repro_dfs:
            final_repro_df = pd.concat(all_repro_dfs)
            
            # The critical de-duplication step using the temporary key
            final_repro_df.drop_duplicates(subset=['uq_map_name', 'noise_level_id'], keep='first', inplace=True)
            
            # Clean up the dataframe for saving
            final_repro_df.drop(columns=['noise_level_id', 'is_ood'], inplace=True)
            final_repro_df.rename(columns=AGGREGATOR_NAME_MAPPING, inplace=True)
            
            base_name = f'{args.real_task}_{args.dataset_name}_{args.variation}_{args.decomp}'
            if args.spatial: base_name += f'_{args.spatial}'
            repro_path = paths.output.joinpath(f'tables/eaurc_reproducibility_repo/{base_name}.csv')
            final_repro_df.to_csv(repro_path, index=False)
            print(f"\nComprehensive and de-duplicated reproducibility data saved to {repro_path}")

def main():
    # Set up plot style
    setup_plot_style_aurc()
    
    # Parse arguments 
    args = parse_args()
    
    if args.label_path is None:
        args.label_path = args.uq_path 
    
    if 'softmax' in args.uq_methods and args.decomp != 'pu':
        raise ValueError('Softmax uncertainty maps cannot be decomposed')
    
    #Set paths and make sure output directory exists
    paths = setup_paths(args)
    
    # Handle panoptic task selection for Lizard data
    if args.variation.startswith('glas_set'):
        variation_1, variation_2, real_task = args.variation.split('_')
        args.variation = f"{variation_1}_{variation_2}"
        args.real_task = 'semantic' if real_task == 'sem' else 'instance'
    elif args.variation.startswith('blood_cells'):
        args.real_task = 'semantic'
    elif args.variation.startswith('nuclei_intensity'):
        args.real_task = 'instance'
    else:
        args.real_task = args.task
    
    # Clean Excel file for plot
    clear_csv_file(paths.output, args)
    
    # Run evaluation
    run_aurc_evaluation(args, paths)

if __name__ == "__main__":
    main()