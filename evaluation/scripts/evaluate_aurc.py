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

from evaluation.constants import (CLASS_NAMES_ARCTIQUE, 
                       CLASS_NAMES_LIZARD, 
                       AUROC_STRATEGIES, 
                       BACKGROUND_FREE_STRATEGIES, 
                       COLORS)
from evaluation.metrics.selective_risk_coverage import compute_selective_risks_coverage
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
    # Check sleective risk-classification file's existence 
    csv_file = output_path.joinpath(
        f'tables/aurc_{args.data_mod}/aurc_data_{args.aggregator_type}_aggr_multi_uq_methods_{args.task}_{args.variation}_id.csv'
    )
    # Ensure directory exists
    csv_file.parent.mkdir(exist_ok=True, parents=True)
    
    if csv_file.exists():
        csv_file.open('w').close()  # Open in write mode to clear contents
        print(f"Cleared content of {csv_file}")
    else:
        print(f"{csv_file} does not exist yet.")
        
    if args.variation == 'blood_cells':
        task = 'semantic' 
    elif args.variation == 'nuclei_intensity':
        task = 'instance'
    else:
        task = args.task
        
    # Check aurc barplots file's existence
    aurc_csv_name = f'{task}_{args.dataset_name}_{args.variation}_{args.decomp}'
    if args.spatial: 
        aurc_csv_name += f'_{args.spatial}'
    aurc_csv_file = output_path.joinpath(f'tables/aurc_{args.data_mod}/{aurc_csv_name}_aurc_{args.data_mod}_results.csv')
    
    if aurc_csv_file.exists():
        aurc_csv_file.open('w').close()  # Open in write mode to clear contents
        print(f"Cleared content of {aurc_csv_file}")
    else:
        print(f"{aurc_csv_file} does not exist yet.")
    
    # Check aurc barplots file's existence
    eaurc_csv_name = f'{task}_{args.dataset_name}_{args.variation}_{args.decomp}'
    if args.spatial: 
        eaurc_csv_name += f'_{args.spatial}'
    eaurc_csv_file = output_path.joinpath(f'tables/eaurc_{args.data_mod}/{aurc_csv_name}_eaurc_{args.data_mod}_results.csv')
    
    if eaurc_csv_file.exists():
        eaurc_csv_file.open('w').close()  # Open in write mode to clear contents
        print(f"Cleared content of {eaurc_csv_file}")
    else:
        print(f"{eaurc_csv_file} does not exist yet.")

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
    decomp = args.decomp
    aggregator_type = args.aggregator_type
    num_workers = args.num_workers
    dataset_name = args.dataset_name
    variation = args.variation #if args.variation else 'LizardData'
    real_task = None
    
    # Handle Lizard case in which variation is the same for both instance and semantic task and Arctique case in which variation is different between tasks
    if variation.startswith('glas_set'):
        variation_1, variation_2, real_task = variation.split('_')
        args.variation = f"{variation_1}_{variation_2}"
        real_task = 'semantic' if real_task == 'sem' else 'instance'
    elif variation.startswith('blood_cells'):
        real_task = 'semantic'
    elif variation.startswith('nuclei_intensity'):
        real_task = 'instance'
    
    # Define extra variables useful for evaluating aurc and saving files later
    ood = (args.data_mod == 'ood')
    return_one_only = True if args.data_mod != 'id_ood' else False
    
    # define parameters along which to loop
    noise_levels = [noise.strip() for noise in args.image_noise.split(',')]
    # if len(noise_levels)>1: # We will now treat the case of aurc for evaluation on both id and ood data
    #     warnings.warn(
    #         "Select only one noise level for this downstream task. "
    #     "Proceeding with the automatic selection based on the argument 'data_mod'..."
    #     )
    #     noise_levels = noise_levels[-1] if ood is True else noise_levels[0]
        
    uq_methods = [uq.strip() for uq in args.uq_methods.split(',')]
    
    # Select appropriate strategies based on aggregator type and extract method names for plotting
    strategies, method_names = select_strategies(aggregator_type)
    
    # This makes it available to both the computation and plotting functions.
    strategies.setdefault('Spatial', {})['GMM'] = (None, None)
    
    # This ensures method_names includes 'GMM' and is in the correct order.
    method_names = []
    for category, methods in strategies.items():
        method_names.extend(methods.keys())
        
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
        'real_task' : real_task,
    }

    concatenated_data = load_dataset_abstract_class(
        paths=paths, 
        image_noises=noise_levels,
        num_workers=1,
        extra_info=extra_info,
        dataset_name=args.dataset_name,
        task=args.task,
        return_one_only=return_one_only,
        uq_methods=uq_methods,
        ood=ood,
    )
        
    # Store results for all methods
    all_results = {
        "aurc": [],
        "eaurc": [],
        "coverages": None,
        "selective_risks": []
    }
    
    # Extract combo keys from concatenated_data
    first_uq_method = next(iter(concatenated_data.keys()))
    combo_keys = list(concatenated_data[first_uq_method].keys())
    
    print(f"Processing combo keys: {combo_keys}")
    
    
    for combo_key in combo_keys:
        # Convert concatenated data to cached maps format
        cached_maps = create_cached_maps_from_concatenated(concatenated_data, combo_key, args, real_task)
        task = args.task
        print(f"Evaluating {task} task")
        
        for idx, uq_method in enumerate(list(cached_maps.keys())):
            print(f"\n=== Processing UQ method: {uq_method} ===")
            masks = cached_maps[uq_method]['real_masks']
            preds = cached_maps[uq_method]['masks']
            uq_maps = cached_maps[uq_method]['maps']
            sample_names = cached_maps[uq_method]['sample_names']
            gt_labels = cached_maps[uq_method]['gt_labels']
                        
            if dataset_name.startswith('arctique') and return_one_only is True:
                # Overleay colours 
                label_colors_sem = {
                    0: [0, 0, 0],             # Background - black or transparent
                    1: [102, 0, 153],         # Epithelial - deep purple
                    2: [0, 0, 255],           # Plasma Cells - blue
                    3: [255, 255, 0],         # Lymphocytes - yellow
                    4: [255, 105, 180],       # Eosinophils - reddish pink
                    5: [0, 255, 0],           # Fibroblasts - green
                }
                
                # Overleay colours 
                label_colors_inst = {
                    0: [0, 0, 0],             # Background - black or transparent
                    1: [255, 105, 180],       # Border - reddish pink
                    2: [0, 0, 0],             # Nucleus - black or transparent
                }
                
                def label_to_rgb(label_map, label_colors):
                    """Converts a (H, W) label map to an (H, W, 3) RGB overlay."""
                    h, w = label_map.shape
                    rgb = np.zeros((h, w, 3), dtype=np.uint8)
                    for label, color in label_colors.items():
                        mask = (label_map == label)
                        rgb[mask] = color
                    return rgb

                # Main visualization
                
                if task.startswith('semantic'):
                    mask = masks[0][...,1]  # (H, W)
                    prediction = preds[0][...,1]  # (H, W)
                    label_colors = label_colors_sem        
                else:
                    mask = masks[0][...,2] # (H, W) 
                    prediction = preds[0][...,2]  # (H, W)
                    label_colors = label_colors_inst
                    
                uq_map = uq_maps[0].array

                # Generate colored overlays
                mask_rgb = label_to_rgb(mask, label_colors)
                pred_rgb = label_to_rgb(prediction, label_colors)

                # Create subplots
                fig, axs = plt.subplots(1, 3, figsize=(16, 5))
                titles = ['Ground Truth', 'Prediction', 'UQ Map']
                imgs = [mask_rgb, pred_rgb, uq_map]

                for ax, title, img in zip(axs, titles, imgs):
                    if title in ['Ground Truth', 'Prediction']:
                        ax.imshow(img)
                    elif title == 'UQ Map':
                        ax.imshow(img, cmap='inferno')
                    ax.set_title(title, fontsize=10)
                    ax.axis('off')

                fig.suptitle(f"Sample", fontsize=12)
                plt.tight_layout()
                plt.subplots_adjust(top=0.85)

                output_dir = Path(__file__).parent
                output_file = output_dir / f'debugging_{task}_sample_plot.png'
                plt.savefig(output_file, bbox_inches='tight')
                plt.close()
                print(f"Overlay plot saved to {output_file}")
                        
            # Analyze uncertainty and generate results
            print(f"Analyzing uncertainty using {aggregator_type} aggregation strategies with {uq_method}")
            results = compute_selective_risks_coverage(
                uq_maps,
                masks,
                preds,
                sample_names, 
                gt_labels,
                paths,
                task,
                model_noise,
                uq_method,
                decomp,
                args.variation,
                noise_levels,
                strategies,
                num_workers,
                dataset_name,
                ood,
                return_one_only,
            )
            
            # Store results
            all_results["coverages"] = results["coverages"]
            all_results["selective_risks"].append(results["selective_risks"])
            all_results["aurc"].append(results["aurc"])
            all_results["eaurc"].append(results["eaurc"])

        # Calculate mean and std across all UQ methods
        mean_aurc = np.mean(np.array(all_results["aurc"]), axis=0)
        std_aurc = np.std(np.array(all_results["aurc"]), axis=0)
        mean_eaurc =  np.mean(np.array(all_results["eaurc"]), axis=0)
        std_eaurc = np.std(np.array(all_results["eaurc"]), axis=0)
        mean_selective_risks = np.mean(np.array(all_results["selective_risks"]), axis=0)
        std_selective_risks = np.std(np.array(all_results["selective_risks"]), axis=0)
        
        # Create final results structure for plotting
        final_results = AnalysisResults(
            mean_aurc=mean_aurc,
            std_aurc=std_aurc,
            mean_eaurc=mean_eaurc,
            std_eaurc=std_eaurc,
            coverages=all_results["coverages"],
            mean_selective_risks=mean_selective_risks,
            std_selective_risks=std_selective_risks
        )
        
        # Create plot
        create_selective_risks_coverage_plot(method_names, final_results, paths.output, args, ood)

def main():
    # Set up plot style
    setup_plot_style_aurc()
    
    # Parse arguments 
    args = parse_args()
    
    if args.label_path is None:
        args.label_path = args.uq_path 
    
    if 'softmax' in args.uq_methods and args.decomp != 'pu':
        raise ValueError('Softmax uncertainty maps cannot be decomposed')
        
    # if not args.variation:
    #     alt_names = variation_name()
    #     args.variation = alt_names[args.dataset_name]
    
    #Set paths and make sure output directory exists
    paths = setup_paths(args)
    
    # Clean Excel file for plot
    clear_csv_file(paths.output, args)
    
    # Run evaluation
    run_aurc_evaluation(args, paths)

if __name__ == "__main__":
    main()