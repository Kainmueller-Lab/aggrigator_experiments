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

from datasets.ADE20K.ade20k_loader import ADE20K
from datasets.Arctique.arctique_dataset_creation import ArctiqueDataset
from datasets.LIDC.lidc_dataset_creation import LIDCDataset
from datasets.Weedsgalore.weedsgalore_dataset_creation import weedsgalore_dataset
from datasets.Lizard.lizard_dataset_creation import LizardDataset
from datasets.GTA_CityScapes.gta_cityscapes_dataset_creation import GTA_CityscapesDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create correlation matrix for aggregation strategies evaluated on a dataset')
    parser.add_argument('--dataset_config', type=str, default='evaluation/configs/ade20k_deeplabv3.yaml', help='Path to config file')
    parser.add_argument('--sample_size', type=int, default='0', help='Number of samples from dataset used to evaluate correlation matrix. If 0, all samples are used.')
    parser.add_argument('--num_workers', type=int, default='16', help='Number of workers for parallel processing. If 0, all available CPUs are used.')
    args = parser.parse_args()

    DATASET = "arctique"
    
    if DATASET == "ade20k":
        config = load_dataset_config(args.dataset_config)
        dataset = ADE20K(config['image_dir'],
                        config['label_dir'],
                        config['uq_map_dir'],
                        config['prediction_dir'],
                        config['metadata_dir'])
        dataset.num_classes = 150
        evaluate_spatial_fingerprint(dataset, args.sample_size, args.num_workers)
    

    if DATASET == "arctique":
        extra_info = {
            'task' : 'semantic',
            'variation' : 'blood_cells',
            'model_noise' : 0,
            'data_noise': '0_00',
            'uq_method' : 'dropout',
            'decomp' : 'pu',
            'spatial' : 'high_moran',
            'metadata' : True,
        }
        
        main_folder_name = "UQ_maps" if not extra_info['spatial'] else "UQ_spatial"
        map_path = Path('/fast/AG_Kainmueller/vguarin/hovernext_trained_models/trained_on_cluster/uncertainty_arctique_v1-0-corrected_14')
        base_path = Path('/fast/AG_Kainmueller/synth_unc_models/data/v1-0-variations/variations/')
        
        image_path = base_path.joinpath(extra_info['variation'], extra_info['data_noise'], 'images')
        mask_path = base_path.joinpath(extra_info['variation'], extra_info['data_noise'], 'masks')
        prediction_path = map_path.joinpath('UQ_predictions')
        uq_map_path = map_path.joinpath(main_folder_name)
        
        dataset = ArctiqueDataset(image_path, 
                                    mask_path, 
                                    uq_map_path, 
                                    prediction_path, 
                                    'abc',
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
        spatial = False
        main_folder_name = "UQ_maps" if not spatial else "UQ_spatial"
        base_path = Path('/fast/AG_Kainmueller/data/ValUES/')
        map_path = base_path
        
        extra_info = {
            'task' : 'fgbg',
            'variation' : 'malignancy', # 'malignancy' or 'texture'
            'model_noise' : 0,
            'data_noise': '0_00', # '0_00' or '1_00'
            'uq_method' : 'dropout',
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
            'uq_method' : 'dropout',
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
        image_path = "/fast/AG_Kainmueller/data/GTA/CityScapesOriginalData/preprocessed/images/"
        mask_path = "/fast/AG_Kainmueller/data/GTA/CityScapesOriginalData/preprocessed/labels/"
        uq_map_path = "/fast/AG_Kainmueller/data/GTA_CityScapes_UQ/Dropout-Final/test_results/fold0_seed123/ood/pred_entropy/"
        prediction_path = "/fast/AG_Kainmueller/data/GTA_CityScapes_UQ/Dropout-Final/test_results/fold0_seed123/ood/pred_seg/"

        dataset = GTA_CityscapesDataset(image_path=image_path, 
                             mask_path= mask_path, 
                             uq_map_path=uq_map_path, 
                             prediction_path=prediction_path, 
                             semantic_mapping_path="")
        dataset_name = "cityscapes_dropout_pu"
        dataset.num_classes = None
        evaluate_spatial_fingerprint(dataset, args.sample_size, args.num_workers, dataset_name)
        


