import argparse
import os
import numpy as np
from pathlib import Path

from evaluation.data_utils import (
    setup_paths, 
    load_dataset_abstract_class, 
    create_cached_maps_from_concatenated
)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Storing subsamples of data used in downstream tasks')
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
    # weedsgalore: '/fast/AG_Kainmueller/data/UQ_maps/weedsgalore/'
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
        '--model_checkpoint', type=str, default=None,
        help='Pretrained model to pass to extra_info[metadata][model_checkpoint]'
    )
    # ade20k: 'deeplabv3_r50-d8_4xb4-160k_ade20k-512x512'
    parser.add_argument(
        '--data_mod', type=str, default='id', 
        help='Data Modality (e.g. ood, id or id_ood)'
    )
    return parser.parse_args()

def main():
    # Parse arguments 
    args = parse_arguments()
            
    if args.label_path is None:
        args.label_path = args.uq_path 
        
    if args.spatial and args.decomp != 'pu':
        raise ValueError('Spatially weighted uncertainty maps calculated only for total predictive uncertainty')
    
    # define parameters along which to loop
    noise_levels = [noise.strip() for noise in args.image_noise.split(',')]
    uq_methods = [uq.strip() for uq in args.uq_methods.split(',')]
    real_task = args.task
    
    # Define extra variables useful for evaluating each noise separataely
    ood = (args.data_mod == 'ood')
    
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
        'real_task' : args.task,
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
        uq_methods=uq_methods,
        return_one_only=True,
        ood=ood,
    )
        
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
            images = cached_maps[uq_method]['input']
            
            # Extract array from Aggrigator obj. UncertaintyMap
            uq_maps = [u.array for u in uq_maps]
            
            # Define the base output directory
            base_output_dir = Path('/fast/AG_Kainmueller/data/AggroUQ_eval_samples_vis')

            # Construct the full path according to the specified structure
            output_path = base_output_dir / args.dataset_name / f"{args.variation}_{combo_key}" / task / uq_method
            
            # Define the final data-specific directories
            decomp_dir = output_path / args.decomp
            pred_dir = output_path / "pred"
            input_dir = output_path / "input"
            gt_seg_dir = output_path / "gt_seg"
            
            # Create all directories, including parents, if they don't exist
            os.makedirs(decomp_dir, exist_ok=True)
            os.makedirs(pred_dir, exist_ok=True)
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(gt_seg_dir, exist_ok=True)

            print(f"Saving files to: {output_path}")

            # Loop through all samples and save the corresponding data
            for i, sample_name in enumerate(sample_names):
                file_name = f"{sample_name}.npy"
                # Save the UQ map
                np.save(decomp_dir / file_name, uq_maps[i])
                # Save the prediction
                np.save(pred_dir / file_name, preds[i])
                # Save the ground truth segmentation mask
                np.save(gt_seg_dir / file_name, masks[i])
                # Save the input image
                np.save(input_dir / file_name, images[i])
            
            print(f"Finished saving {len(sample_names)} samples for UQ method: {uq_method}")

if __name__ == "__main__":
    main()
            
            
if __name__ == "__main__":
    main()