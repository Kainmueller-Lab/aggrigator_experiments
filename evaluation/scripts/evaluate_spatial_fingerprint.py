import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import multiprocessing
import yaml
import time

from joblib import Parallel, delayed

from aggrigator.uncertainty_maps import UncertaintyMap
from aggrigator.spatial_decomposition import spatial_decomposition # NOTE: This is only available on the develop branch of the aggrigator repo. Use "pip install -e ." to install the package.



def load_dataset_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config





def evaluate_spatial_fingerprint(dataset, sample_size, num_workers):
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
    print(f"Number of samples used for spatial fingerprint: {sample_size} of {len(dataset)}")
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
        uq_array = uq_array / np.log(dataset_info['num_classes'])

        # Compute spatial decomposition for all spatial measures
        spatial_measures = ["moran", "entropy", "eds"]
        window_size = 3
        uq_map = UncertaintyMap(array=uq_array, mask=None, name=sample_name)
        measure_mass_ratios = {measure: spatial_decomposition(uq_map, window_size=window_size, spatial_measure=measure)[3] for measure in spatial_measures}
        return (sample_name, measure_mass_ratios)

    # Decompose all UQ maps
    start = time.time()
    n_jobs = 8 if num_workers == 0 else num_workers # NOTE: Strangely this gets slower for larger num_workers.
    measure_mass_ratios = Parallel(n_jobs=n_jobs, verbose=10)(delayed(get_measure_mass_ratios)(dataset[idx]) for idx in range(sample_size))
    measure_mass_ratio_df = pd.DataFrame.from_dict(dict(measure_mass_ratios), orient='index')
    print(f"Computed spatial measure mass ratios: {time.time() - start} s")


    # Save to csv
    try:
        dataset_name = dataset.get_info()['dataset_name']
    except:
        dataset_name = "" # NOTE: Please add dataset_name member to dataset class
    out_name = f"spatial_fingerprint_{dataset_name}"
    measure_mass_ratio_df.to_csv(os.path.join("output", "tables", f"{out_name }.csv"))
    print(f"Correlation matrix {out_name}.csv saved to output folder.")








import argparse

#from evaluation.scripts.evaluate_correlation import evaluate_correlation, load_dataset_config
from datasets.ADE20K.ade20k_loader import ADE20K


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create correlation matrix for aggregation strategies evaluated on a dataset')
    parser.add_argument('--dataset_config', type=str, default='configs/ade20k_deeplabv3.yaml', help='Path to config file')
    parser.add_argument('--sample_size', type=int, default='0', help='Number of samples from dataset used to evaluate correlation matrix. If 0, all samples are used.')
    parser.add_argument('--num_workers', type=int, default='0', help='Number of workers for parallel processing. If 0, all available CPUs are used.')
    args = parser.parse_args()
    
    config = load_dataset_config(args.dataset_config)
    
    dataset = ADE20K(config['image_dir'],
                    config['label_dir'],
                    config['uq_map_dir'],
                    config['prediction_dir'],
                    config['metadata_dir'])
    
    evaluate_spatial_fingerprint(dataset, args.sample_size, args.num_workers)

    

