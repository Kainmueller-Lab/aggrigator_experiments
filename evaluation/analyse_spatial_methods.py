import os
import numpy as np
import argparse
from aggrigator.uncertainty_maps import UncertaintyMap
from aggrigator.summary import AggregationSummary
from aggrigator.strategylist import StrategyList

def analyse_spatial_methods(dataset, num_cpus, slice_idx):
    """
    Processes uncertainty maps and applies aggregation strategies.

    :param dataset: String, name of the dataset folder inside 'data'.
    :param num_cpus: Integer, number of CPU cores to use for computation.
    :param slice_idx: Integer, specifying the slice index to extract from 3D arrays.
    """
    # Define dataset path
    path = os.path.join(os.getcwd(), "data", dataset)

    # Load all 3D arrays and extract the specified slice
    files = os.listdir(path)
    arrays = []
    array_dim = len(np.load(os.path.join(path, files[0])).shape)
    if array_dim == 3:
        arrays = [np.load(os.path.join(path, file))[slice_idx, :, :] for file in files]
    else:
        arrays = [np.load(os.path.join(path, file)) for file in files]

    # Ensure all negative values are set to 0 (in case they occur of precision errors)
    arrays = [np.where(array < 0, 0, array) for array in arrays]

    # Create UncertaintyMap objects
    unc_maps = [UncertaintyMap(array=array, name=name) for array, name in zip(arrays, files)]

    # Define strategy list and summary object
    strategy_list = StrategyList.SPATIAL
    suffix = f"_slice_{slice_idx}" if array_dim==3 else ""
    summary = AggregationSummary(strategy_list, name=f"spatial_analysis_{dataset}{suffix}", num_cpus=num_cpus)

    # Apply aggregation methods and save results
    summary.apply_methods(unc_maps, save_to_excel=True, do_plot=False)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process uncertainty maps and apply aggregation strategies.")
    parser.add_argument("--dataset", type=str, default="arctique_uq_maps", help="Dataset folder name inside 'data'")
    parser.add_argument("--num_cpus", type=int, default=12, help="Number of CPU cores to use for computation (default: 12)")
    parser.add_argument("--slice", type=int, default=40, help="Slice index to extract from 3D arrays (default: 25)")

    # Parse arguments
    args = parser.parse_args()

    # Run main function with parsed arguments
    analyse_spatial_methods(args.dataset, args.num_cpus, args.slice)
