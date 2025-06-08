import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from evaluation.data_utils import (DataPaths,
                                   setup_paths, 
                                   load_predictions, 
                                   load_dataset,
                                   load_unc_maps, 
                                   validate_indices,
                                   rescale_maps)
from evaluation.metrics.dice import *

# TODO - to move later in Datasets folder almost as a preprocessing 

# ---- Configuration Functions ----

def variation_name():
    # When there is no clear id and ood distinction in the inputs 
    return { 
        'lizard' : 'LizardData'
    } 

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
    parser.add_argument('--image_noise', type=str, default='0_00,0_25,0_50,0_75,1_00', help='Comma-separated list of image noise to evaluate')
    parser.add_argument('--uq_methods', type=str, default='tta,softmax,ensemble,dropout', help='Comma-separated list of UQ methods to evaluate')
    parser.add_argument('--decomp', type=str, default='pu', help='Decomposition component (e.g. pu, au, eu)')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of workers for parallel processing' )
    parser.add_argument('--dataset_name', type=str, default='arctique', help='Selected dataset (e.g. arctique, lizard, lidc)' )
    
    return parser.parse_args()

# ---- Analysis Functions ----

def compute_error(pred_list, gt_list):
    return np.abs(gt_list - pred_list) 
    
def process_uq_method(args, paths, mask_key, data, noise, uq_method, gt_list_old, samples_num):
    """Process a single UQ method for given noise level and mask type."""
    print(f"\n=== Processing UQ method: {uq_method} ===")
    
    # Load uncertainty maps
    uq_maps = load_unc_maps(
        uq_path=paths.uq_maps, 
        task=args.task, 
        model_noise=args.model_noise, 
        variation=args.variation, 
        data_noise=noise, 
        uq_method=uq_method, 
        decomp=args.decomp, 
        dataset_name=args.dataset_name,
        calibr=(args.dataset_name in ['arctique', 'lizard']),
    )
    
    uq_maps = rescale_maps(uq_maps, uq_method, args.task, args.dataset_name)
    
    # Load prediction data for current UQ method
    pred_list = load_predictions(
        paths,
        args.model_noise,
        args.variation,
        noise,
        uq_method,
        args.dataset_name,
    )
            
    # Load and check metadata indices for consistency
    gt_list = validate_indices(
        args, noise, paths.metadata, uq_method, data, gt_list_old[:samples_num], args.dataset_name
    )
    
    print(f"\n--- Processing gt_list with {len(gt_list)} samples ---")

    # Generate error maps and compute uncertainty-error continous dice score
    err_maps = compute_error(pred_list, gt_list)
    unc_dice = continuous_dice_coefficient(uq_maps, err_maps)
    threshold = .5
    uq_maps_binary = (uq_maps >= threshold).astype(np.uint8)
    
    # Compute continous and discrete dice between uncertainty and error maps
    unc_dice_discrete = dice_coefficient_torchmetrics(uq_maps_binary, err_maps)
    print(np.mean(unc_dice), np.mean(unc_dice_discrete))
    
    # Random index
    idx = np.random.randint(0, len(gt_list))
    uq_example = uq_maps[idx]  # original uncertainty map (not binarized)
    uq_binarized_example = uq_maps_binary[idx]
    pred_example = pred_list[idx]
    gt_example = gt_list[idx]

    # Compute TP, FP, FN
    tp = (pred_example == 1) & (gt_example == 1)
    fp = (pred_example == 1) & (gt_example == 0)
    fn = (pred_example == 0) & (gt_example == 1)

    # Create RGB overlay image
    overlay = np.zeros((*pred_example.shape, 3), dtype=np.uint8)
    overlay[tp] = [0, 255, 0]    # Green for TP
    overlay[fp] = [255, 0, 0]    # Red for FP
    overlay[fn] = [0, 0, 255]    # Blue for FN

    # Plotting with GridSpec
    fig = plt.figure(figsize=(14, 4))
    gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 1.1])  # last plot slightly wider for colorbar

    # Overlay map
    ax0 = plt.subplot(gs[0])
    ax0.imshow(overlay)
    ax0.set_title("TP (Green) / FP (Red) / FN (Blue)", fontsize=10)
    ax0.axis('off')

    # Prediction
    ax1 = plt.subplot(gs[1])
    ax1.imshow(pred_example, cmap='gray')
    ax1.set_title("Prediction", fontsize=10)
    ax1.axis('off')

    # Ground Truth
    ax2 = plt.subplot(gs[2])
    ax2.imshow(gt_example, cmap='gray')
    ax2.set_title("Ground Truth", fontsize=10)
    ax2.axis('off')
    
    # Uncertainty heatmap with colorbar
    ax3 = plt.subplot(gs[3])
    ax3.imshow(uq_binarized_example, cmap='gray')
    ax3.set_title("Binarized Uncertainty Map", fontsize=10)
    ax3.axis('off')

    # Uncertainty heatmap with colorbar
    ax4 = plt.subplot(gs[4])
    im = ax4.imshow(uq_example, cmap='hot')
    ax4.set_title("Uncertainty Map", fontsize=10)
    ax4.axis('off')
    cbar = fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label("Uncertainty")

    # Add figure-wide title
    fig.suptitle(f"Continuous Dice: {unc_dice_discrete[idx]:.3f} | Discrete Dice: {unc_dice_discrete[idx]:.3f}", fontsize=10, y=1.05)
    
    # Save
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    filepath = f'/fast/AG_Kainmueller/vguarin/aggrigator_experiments/evaluation/scripts/{uq_method}_{args.variation}_{mask_key}_uq_visualization.png'
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    
    # return results if needed


def compute_uncertainty_dice(args: argparse.Namespace, paths: DataPaths) -> None:
    """
    Compute uncertainty dice scores for different noise levels and UQ methods.
    
    Args:
        args: Command line arguments
        paths: Data paths configuration
    """
    
    # Extract parameters from arguments
    image_noise = args.image_noise.split(',')
    uq_methods = args.uq_methods.split(',')
    
    print(f"Image noise levels: {image_noise}")
    print(f"UQ methods: {uq_methods}")
       
    # Load concatenated iD and OoD dataset and ground truth
    dataset, gt_list_old, _ = load_dataset(
        data_path=paths.data, 
        image_noise=image_noise,
        num_workers=args.num_workers,
        dataset_name=args.dataset_name,
    )
    
    # Process each mask type in the dataset
    for mask_key, data in dataset.items():
        samples_num = len(data.dataset)
        print(f"\n--- Processing {mask_key} with {samples_num} samples ---")
        
        # Determine noise levels to process based on mask type
        if mask_key == 'id_masks':
            # For ID masks, only process the first noise level
            noise_levels = [image_noise[0]]
        else:
            # For OOD masks, process all noise levels except the first
            noise_levels = image_noise[1:]
        
        # Process each noise level and UQ method combination
        for noise in noise_levels:
            print(f"\n-- Processing noise level: {noise} --")
            
            for uq_method in uq_methods:
                process_uq_method(
                    args, paths, mask_key, data, noise, uq_method, gt_list_old, samples_num
                )
    
def main():    
    # Parse arguments 
    args = parse_args()
    if args.label_path is None:
        args.label_path = args.uq_path 
    if not args.variation:
        alt_names = variation_name()
        args.variation = alt_names[args.dataset_name]
    
    #Set paths and make sure output directory exists
    paths = setup_paths(args)
        
    # Run computation
    compute_uncertainty_dice(args, paths)

if __name__ == "__main__":
    main()