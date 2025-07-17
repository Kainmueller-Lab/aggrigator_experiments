import argparse
import numpy as np
import pandas as pd
import os
import yaml
import time

from pathlib import Path
from joblib import Parallel, delayed

from aggrigator.uncertainty_maps import UncertaintyMap
from aggrigator.spatial_decomposition import spatial_decomposition # NOTE: This is only available on the develop branch of the aggrigator repo. Use "pip install -e ." to install the package.



def load_dataset_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config





def evaluate_spatial_fingerprint(dataset, sample_size, num_workers, dataset_name=None):
    """
    Evaluate the spatial fingerprint of the given dataset:
    For each spatial measure, compute the uncertainty mass ratio for each uncertainty map.
    Store final result in a pandas DataFrame.
    Plot the results as violin plots, illustrating where on the spatial spectrum the dataset is concentrated.

    Args:
        dataset (Dataset_Class): Dataset object.
        sample_size (int): Number of samples to use for evaluation.
        num_workers (int): Number of workers for parallel processing.
    """
    sample_size = len(dataset) if sample_size == 0 else sample_size

    # Print info
    dataset_info = dataset.get_info()
    dataset_info.pop('semantic_mapping') # NOTE: Semantic mapping too long in case of many classes
    print("____________________")
    print(f"Evaluating spatial fingerprint")
    for key, value in dataset_info.items():
        print(f"{key}: {value}")
    if dataset_name is not None:
        print(f"Dataset name: {dataset_name}")
    print(f"Number of samples used for spatial fingerprint: {sample_size} of {len(dataset)}")
    # This is an ugly hack. In future, make sure that dataset.num_classes is defined.
    if dataset.num_classes is None:
        print(f"WARNING: Could not normalize UQ maps because dataset_info['num_classes'] or dataset.num_classes is not defined.")
    else:
        print(f"NOTE: Normalizing UQ maps by ln(K) where K={dataset.num_classes} is the number of classes.")
    print("____________________")

    def get_measure_mass_ratios(sample):
        # Load uncertainty maps and masks from dataset
        mask = sample['mask']
        uq_array = sample['uq_map']
        sample_name = sample['sample_name']

        # Slice if 3D
        if uq_array.ndim == 3:
            mid_slice = uq_array.shape[0] // 2
            uq_array = uq_array[mid_slice, :, :]
            mask = mask[mid_slice, :, :]
        
        # Replace negative values with zero
        # NOTE: Such values (close to zero) sometimes occur and need to be dealt with.
        uq_array = np.where(uq_array < 0, 0, uq_array)
        
        # Normalize arrays by ln(K) where K is number of classes if UQ maps are not normalized in dataloader
        if dataset.num_classes is not None:
            uq_array = uq_array / np.log(dataset.num_classes) 
        # uq_array = uq_array / np.log(dataset_info['num_classes'])

        # Compute spatial decomposition for all spatial measures
        spatial_measures = ["moran", "entropy", "eds"]
        window_size = 3
        uq_map = UncertaintyMap(array=uq_array, mask=None, name=sample_name)
        measure_mass_ratios = {measure: spatial_decomposition(uq_map, window_size=window_size, spatial_measure=measure)[3] for measure in spatial_measures}
        return (sample_name, measure_mass_ratios)

    # Decompose all UQ maps
    start = time.time()
    n_jobs = 16 if num_workers == 0 else num_workers # NOTE: Strangely this gets slower for larger num_workers.
    #measure_mass_ratios = [get_measure_mass_ratios(dataset[idx]) for idx in range(sample_size)]
    measure_mass_ratios = Parallel(n_jobs=n_jobs, verbose=10)(delayed(get_measure_mass_ratios)(dataset[idx]) for idx in range(sample_size))
    measure_mass_ratio_df = pd.DataFrame.from_dict(dict(measure_mass_ratios), orient='index')
    print(f"Computed spatial measure mass ratios: {time.time() - start} s")


    # Save to csv
    if dataset_name is None:
        try:
            dataset_name = dataset.get_info()['dataset_name']
        except:
            dataset_name = "" # NOTE: Please add dataset_name member to dataset class
    out_name = f"spatial_fingerprint_{dataset_name}"
    measure_mass_ratio_df.to_csv(os.path.join("output", "tables", f"{out_name }.csv"))
    print(f"Spatial fingerprint {out_name}.csv saved to output folder.")








import argparse

from datasets.ADE20K.ade20k_dataset_creation import ADE20K_CityscapesDataset
from datasets.Arctique.arctique_dataset_creation import OptimizedArctiqueDataset, SharedMaskCache
from datasets.LIDC.lidc_dataset_creation import LIDCDataset
from datasets.Weedsgalore.weedsgalore_dataset_creation import weedsgalore_dataset
from datasets.Lizard.lizard_dataset_creation import LizardDataset
from datasets.GTA_CityScapes.gta_cityscapes_dataset_creation import GTA_CityscapesDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create correlation matrix for aggregation strategies evaluated on a dataset')
    parser.add_argument('--dataset', type=str, help='Name of dataset to evaluate correlations. Options: ade20k, arctique, lidc, lizard, cityscapes, weedsgalore')
    parser.add_argument('--uq_method', type=str, help='Name of UQ method used. Options: dropout, softmax')
    parser.add_argument('--sample_size', type=int, default='0', help='Number of samples from dataset used to evaluate correlation matrix. If 0, all samples are used.')
    parser.add_argument('--num_workers', type=int, default='16', help='Number of workers for parallel processing. If 0, all available CPUs are used.')
    args = parser.parse_args()

    DATASET = args.dataset
    
    if DATASET == "ade20k":
        for model_name in ['deeplabv3', 'resnest']:
            noise_levels = ['0_00', '1_00']
            folder  = ['validation', 'test_cityscapes']
            for nl, fold in zip(noise_levels, folder):
            
                model_id = "deeplabv3_r50-d8_4xb4-160k_ade20k-512x512"
                # model_id = "resnest_s101-d8_fcn_4xb4-160k_ade20k-512x512" if model_name == "resnest" else "deeplabv3_r50-d8_4xb4-160k_ade20k-512x512"
                text_path = f"/fast/AG_Kainmueller/data/GTA_ValUES_splits/ADE20k_id_test" 
                
                extra_info = {
                    'task' : 'semantic',
                    'variation' : 'cityscapes',
                    'model_noise' : 0,
                    'data_noise': nl,
                    'uq_method': args.uq_method,
                    'decomp' : 'pu',
                    'spatial' : None,
                    'split_path' : None, #text_path,
                    'split' : None,
                    'metadata' : False,
                    'model_checkpoint': 'deeplabv3_r50-d8_4xb4-160k_ade20k-512x512',
                }
                
                image_path = f'/fast/AG_Kainmueller/data/ADEChallengeData2016/images/{fold}'
                mask_path = f'/fast/AG_Kainmueller/data/ADEChallengeData2016/annotations/{fold}'
                uq_map_path = f'/fast/AG_Kainmueller/data/UQ_maps/ADE20K/'
                prediction_path = '/fast/AG_Kainmueller/data/ADEChallengeData2016/'
                metadata_dir = '/fast/AG_Kainmueller/data/ADEChallengeData2016/objectInfo150.json'
                    
                dataset = ADE20K_CityscapesDataset(image_path, 
                                                        mask_path, 
                                                        uq_map_path, 
                                                        prediction_path, 
                                                        '/fast/AG_Kainmueller/data/ADEChallengeData2016/objectInfo150.json',
                                                        **extra_info)
                dataset.num_classes = 150
                dataset_name = f"ade20k_{model_name}_{extra_info['task']}_{extra_info['variation']}_{extra_info['data_noise']}_{extra_info['uq_method']}_{extra_info['decomp']}"
                evaluate_spatial_fingerprint(dataset, args.sample_size, args.num_workers, dataset_name)
    

    if DATASET == "arctique":
        for task in ['instance', 'semantic']:
            noise_levels = ['0_00', '0_25', '0_50', '0_75', '1_00']
            for noise_level in noise_levels:
                variation = 'blood_cells' if task == 'semantic' else 'nuclei_intensity'
                extra_info = {
                    'task' : task,
                    'variation' : variation,
                    'model_noise' : 0,
                    'data_noise': noise_level,
                    'uq_method' : args.uq_method,
                    'decomp' : 'pu',
                    'spatial' : False,
                    'metadata' : False,
                }
                
                main_folder_name = "UQ_maps" if not extra_info['spatial'] else "UQ_spatial"
                map_path = Path('/fast/AG_Kainmueller/vguarin/hovernext_trained_models/trained_on_cluster/uncertainty_arctique_v1-0-corrected_14')
                base_path = Path('/fast/AG_Kainmueller/synth_unc_models/data/v1-0-variations/variations/')
                
                image_path = base_path.joinpath(extra_info['variation'], extra_info['data_noise'], 'images')
                mask_path = base_path.joinpath(extra_info['variation'], extra_info['data_noise'], 'masks')
                prediction_path = map_path.joinpath('UQ_predictions')
                uq_map_path = map_path.joinpath(main_folder_name)

                mask_cache = SharedMaskCache()
                ref_mask_path = base_path.joinpath(extra_info['variation'], '0_00', 'masks')
                ref_image_path = base_path.joinpath(extra_info['variation'], '0_00', 'images')

                sample_names = [int(digits) for filename in os.listdir(ref_image_path)
                              if (digits := ''.join(filter(str.isdigit, filename)))]
                
                shared_masks = mask_cache.get_masks(ref_mask_path, sample_names, extra_info['task'])
                
                dataset = OptimizedArctiqueDataset(ref_image_path, 
                                            ref_mask_path, 
                                            uq_map_path, 
                                            prediction_path, 
                                            'abc',
                                            shared_masks,
                                            **extra_info)
                dataset_name = f"arctique_{extra_info['task']}_{extra_info['variation']}_{extra_info['data_noise']}_{extra_info['uq_method']}_{extra_info['decomp']}"
                dataset.num_classes = 6
                evaluate_spatial_fingerprint(dataset, args.sample_size, args.num_workers, dataset_name)


    if DATASET == "weedsgalore":
        image_path = "/fast/AG_Kainmueller/data/weedsgalore/weedsgalore-dataset/"
        uq_folder =  "/fast/AG_Kainmueller/data/UQ_maps/weedsgalore/rgb_train/crops_vs_weed/dropout/pu/"
        pred_folder =  "/fast/AG_Kainmueller/data/UQ_maps/weedsgalore/rgb_train/crops_vs_weed/dropout/pred/"
        metadata_file = "/fast/AG_Kainmueller/data/UQ_maps/weedsgalore/rgb_train/crops_vs_weed/dropout/metadata.json"
        dataset = weedsgalore_dataset(image_path=image_path, 
                                 mask_path=image_path, 
                                 uq_map_path=uq_folder, 
                                 prediction_path=pred_folder, 
                                 semantic_mapping_path="", 
                                 metadata_file = metadata_file)
        dataset_name = f"weedsgalore_dropout_pu"
        evaluate_spatial_fingerprint(dataset, args.sample_size, args.num_workers, dataset_name)



    if DATASET == "lidc":
        for variation in ['malignancy', 'texture']:
            for noise_level in ['0_00', '1_00']:
                spatial = False
                main_folder_name = "UQ_maps" if not spatial else "UQ_spatial"
                base_path = Path('/fast/AG_Kainmueller/data/ValUES/')
                map_path = base_path
                
                extra_info = {
                    'task' : 'fgbg',
                    'variation' : variation,
                    'model_noise' : 0,
                    'data_noise': noise_level,
                    'uq_method' : args.uq_method,
                    'decomp' : 'pu',
                    'spatial' : None,
                    'cons_thresh' : 2,
                    'metadata' : True,
                    'render_2d' : True,
                    'render_ind_masks': False,
                }
                
                # Set up paths based on folder structure
                cycle = 'FirstCycle'
                folder = f"{extra_info['variation']}_fold0_seed123"
                placeholder = "Softmax"
                data_path = base_path.joinpath(f"{cycle}/{placeholder}/test_results/{folder}/") 
                
                if extra_info['data_noise'] == "0_00":
                    data_dir = data_path / "id"
                else:
                    data_dir = data_path / "ood"
                    
                image_path = data_dir / "input"
                mask_path = data_dir / "gt_seg"
                prediction_path = map_path.joinpath('UQ_predictions')
                uq_map_path = map_path.joinpath(main_folder_name)
                
                dataset = LIDCDataset(image_path, 
                                        mask_path, 
                                        uq_map_path, 
                                        prediction_path, 
                                        'abc',
                                        **extra_info)
                dataset_name = f"lidc_{extra_info['task']}_{extra_info['variation']}_{extra_info['data_noise']}_{extra_info['uq_method']}_{extra_info['decomp']}"
                dataset.num_classes = 2
                evaluate_spatial_fingerprint(dataset, args.sample_size, args.num_workers, dataset_name)


    if DATASET == "lizard":
        spatial = False
        main_folder_name = "UQ_maps" if not spatial else "UQ_spatial"
        lmdb_path = '/fast/AG_Kainmueller/data/Lizard/lizard_lmdb/'
        # base_path = Path('/fast/AG_Kainmueller/synth_unc_models/data/v1-0-variations/variations/')
        extra_info = {
            'task' : 'instance',
            'variation' : 'glas',
            'model_noise' : 0,
            'data_noise': '0_00',
            'uq_method' : args.uq_method,
            'decomp' : 'pu',
            'spatial' : None,
            'metadata' : True,
            'split_path' : None,
            'split' : ['test']
        }
        
        csv_path = Path(lmdb_path).parent.joinpath(f"splits/domain_shift_splits/lizard_domaingen_{extra_info['variation']}_test_split.csv")
        extra_info['split_path'] = csv_path
        
        dataset = LizardDataset(lmdb_path, 
                                    lmdb_path, 
                                    lmdb_path, 
                                    lmdb_path, 
                                    'abc',
                                    **extra_info)
        dataset_name = f"lizard_{extra_info['task']}_{extra_info['variation']}_{extra_info['data_noise']}_{extra_info['uq_method']}_{extra_info['decomp']}"
        dataset.num_classes = 7
        evaluate_spatial_fingerprint(dataset, args.sample_size, args.num_workers, dataset_name)


    if DATASET == "cityscapes":
        extra_info = {
            'task' : 'semantic',
            'variation' : 'cityscapes',
            'model_noise' : 0,
            'data_noise': '1_00',
            'uq_method': args.uq_method,
            'decomp' : 'pu',
            'spatial' : None,
            'split_path' : None,
            'split' : None
        }

        base_path = "/fast/AG_Kainmueller/data"
        data_folder_name = "/GTA/CityScapesOriginalData" # /GTA/CityScapesOriginalData
        
        if data_folder_name.startswith('/GTA/City'):
            splits_folder = 'Cityscapes_ood'
            
        else:
            splits_folder = 'GTA_id_test'
        
        image_path = f"{base_path}/{data_folder_name}/preprocessed/images/"
        mask_path = f"{base_path}/{data_folder_name}/preprocessed/labels/"
        uq_map_path = f"{base_path}/GTA_CityScapes_UQ/"
        prediction_path = uq_map_path
        
        text_path = f"{base_path}/GTA_ValUES_splits/{splits_folder}"
        extra_info['split_path'] = text_path
        
        dataset = GTA_CityscapesDataset(image_path, 
                                    mask_path, 
                                    uq_map_path, 
                                    prediction_path, 
                                    'abc',
                                    **extra_info)
        dataset_name = f"gta_1_00_{args.uq_method}_pu"
        dataset.num_classes = None
        evaluate_spatial_fingerprint(dataset, args.sample_size, args.num_workers, dataset_name)

    
    if DATASET == "gta":
        image_path = "/fast/AG_Kainmueller/data/GTA/OriginalData/preprocessed/images/" #OriginalData instead of CityScapesOriginalData to evaluate the GTA iD test set
        mask_path = "/fast/AG_Kainmueller/data/GTA/OriginalData/preprocessed/labels/" #OriginalData instead of CityScapesOriginalData to evaluate on GTA iD test set
        uq_map_path = "/fast/AG_Kainmueller/data/GTA_CityScapes_UQ/"
        prediction_path = mask_path

        extra_info = {
                'task' : 'semantic',
                'variation' : 'cityscapes', 
                'model_noise' : 0,
                'data_noise': '0_00', #0_00 for evaluating on GTA iD test set 
                'uq_method': 'dropout',
                'decomp' : 'pu',
                'spatial' : None,
                'split_path' : "/fast/AG_Kainmueller/data/GTA_ValUES_splits/GTA_id_test", # GTA_id_test is the file name for the GTA iD test set samples
                'split' : None
            }

        dataset = GTA_CityscapesDataset(image_path, 
                                        mask_path, 
                                        uq_map_path, 
                                        prediction_path, 
                                        'abc',
                                        **extra_info)
        dataset_name = f"gta_0_00_{args.uq_method}_pu"
        dataset.num_classes = 32
        evaluate_spatial_fingerprint(dataset, args.sample_size, args.num_workers, dataset_name)