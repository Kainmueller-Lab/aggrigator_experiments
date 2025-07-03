import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import multiprocessing
import time
import yaml

from pathlib import Path
from joblib import Parallel, delayed

from aggrigator.uncertainty_maps import UncertaintyMap
from aggrigator.methods import AggregationMethods as am
from aggrigator.summary import AggregationSummary


focus_strategy_list = [
    (am.mean, None),
    (am.above_threshold_mean, 0.3),
    (am.above_threshold_mean, 0.5),
    (am.above_threshold_mean, 0.7),
    (am.above_threshold_mean, 0.9),
    (am.above_threshold_mean, 0.95),
    (am.above_quantile_mean, 0.3),
    (am.above_quantile_mean, 0.5),
    (am.above_quantile_mean, 0.7),
    (am.above_quantile_mean, 0.9),
    #(am.above_quantile_mean, 0.95),
    (am.above_quantile_mean_fg_ratio, None),
    (am.patch_aggregation, 10), 
    (am.patch_aggregation, 20),
    (am.patch_aggregation, 40),
    (am.patch_aggregation, 80),
    (am.patch_aggregation, 100),
    #(am.patch_aggregation, 200),
    (am.class_mean_w_equal_weights, None),
    (am.class_mean_weighted_by_occurrence, None),
]


# TODO: Move to a better utils file.
def load_dataset_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_correlation_matrix_plot(df, filename, save_dir):
    """
    Computes and plots the correlation matrix of methods across columns.

    :param df: pandas DataFrame where each row represents a method and columns represent features.
    """
    # Compute the correlation matrix (rows as methods, columns as features)
    corr_matrix = df[df.columns.tolist()[1:]].T.corr(min_periods=1)

    # Plot the correlation matrix as a heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    strategy_names = df.index.tolist()
    sns.heatmap(corr_matrix, ax=ax, cmap="coolwarm", annot=False, fmt=".2f",
                cbar=True, vmin=-1, vmax=1, xticklabels=strategy_names, yticklabels=strategy_names)
    
    # Color strategy names by category
    color_code = {
        "threshold": "red",
        "quantile": "green",
        "patch": "blue",
        "class_mean": "orange",
    }
    for tick in ax.get_xticklabels():
        strategy_name = tick.get_text()
        color = next((color_code[key] for key in color_code if key in strategy_name), "black")
        tick.set_bbox(dict(facecolor=color, edgecolor='none', alpha=0.5, boxstyle="round,pad=0.3"))
    for tick in ax.get_yticklabels():
        strategy_name = tick.get_text()
        color = next((color_code[key] for key in color_code if key in strategy_name), "black")
        tick.set_bbox(dict(facecolor=color, edgecolor='none', alpha=0.5, boxstyle="round,pad=0.3"))


    plt.title(filename)
    plt.savefig(os.path.join(save_dir, f"{filename}.png"))
    plt.close()


# def to_correlation_matrix(df):
#     method_columns = df.columns.tolist()[1:]
#     corr_matrix = df[method_columns].T.corr(min_periods=1)
#     # Change index and columns names to method_columns
#     corr_matrix.columns = [strat for strat in df["Name"].tolist()]
#     corr_matrix.index = [strat for strat in df["Name"].tolist()]
#     return corr_matrix

def compute_correlations(df):
    method_columns = df.columns.tolist()[1:]
    correlations = {}
    for correlation_type in ["pearson", "spearman", "kendall"]:
        corr_matrix = df[method_columns].T.corr(min_periods=1, method=correlation_type)
        corr_matrix.columns = [strat for strat in df["Name"].tolist()]
        corr_matrix.index = [strat for strat in df["Name"].tolist()]
        correlations[correlation_type] = corr_matrix
    return correlations



def evaluate_correlation(dataset, sample_size, num_workers, dataset_name=None):
    sample_size = len(dataset) if sample_size == 0 else sample_size

    # Print info
    dataset_info = dataset.get_info()
    dataset_info.pop('semantic_mapping') # NOTE: Semantic mapping too long in case of many classes
    print("____________________")
    print(f"Evaluating correlation matrix")
    for key, value in dataset_info.items():
        print(f"{key}: {value}")
    print(f"Number of samples used for correlation matrix: {sample_size} of {len(dataset)}")
    # This is an ugly hack. In future, make sure that dataset.num_classes is defined.
    if dataset.num_classes is None:
        print(f"WARNING: Could not normalize UQ maps because dataset_info['num_classes'] or dataset.num_classes is not defined.")
    else:
        print(f"NOTE: Normalizing UQ maps by ln(K) where K={dataset.num_classes} is the number of classes.")
        if dataset.num_classes != 2 and (am.above_quantile_mean_fg_ratio, None) in focus_strategy_list: # Only apply AQA with FG-BG ratio if there are 2 classes
            focus_strategy_list.pop(focus_strategy_list.index((am.above_quantile_mean_fg_ratio, None)))
    print("____________________")


    def aggregate(sample):
        # Load uncertainty maps and predictions from dataset
        prediction = sample['prediction']
        uq_array = sample['uq_map']

        # NOTE: Weedsgalore prdictions are 3D arrays with a single channel
        if prediction.ndim == 3 and prediction.shape[0] == 1:
            prediction = prediction.squeeze(0)

        # NOTE: Arctique and Lizard predictions are 3D arrays with two channels. 0: instance, 1: 3-class segmentation
        if prediction.ndim == 3 and prediction.shape[2] == 2:
            prediction = prediction[:, :, -1]

        # Slice if 3D
        if uq_array.ndim == 3:
            print(f"Warning: 3D UQ map detected. Only middle 2D slice are used for correlation matrix.")
            mid_slice = uq_array.shape[0] // 2
            uq_array = uq_array[mid_slice, :, :]
            prediction = prediction[mid_slice, :, :]
        
        # Replace negative values with zero
        # NOTE: Such values (close to zero) sometimes occur and need to be dealt with.
        uq_array = np.where(uq_array < 0, 0, uq_array)

        # Ignore too small images bc of patch aggregation with patch size 200
        h, w = uq_array.shape
        patch_200_in_agg_list = (am.patch_aggregation, 200) in focus_strategy_list
        if patch_200_in_agg_list and (h < 200 or w < 200):
            print(f"Warning: Ignoring UQ map {sample['sample_name']} because it is too small for patch aggregation with patch size 200.")
            return None
        
        # Normalize arrays by ln(K) where K is number of classes if UQ maps are not normalized in dataloader
        if dataset_info['num_classes'] is not None:
            uq_array = uq_array / np.log(dataset_info['num_classes'])

        # Apply aggregation strategies
        uq_map = UncertaintyMap(array=uq_array, mask=prediction, name=sample['sample_name'])
        summary = AggregationSummary(focus_strategy_list, num_cpus=1)
        return summary.apply_methods([uq_map], save_to_excel=False, do_plot=False, max_value=1.0)
    
    # Aggregate all UQ maps
    start = time.time()
    n_jobs = multiprocessing.cpu_count() if num_workers == 0 else num_workers
    summary_dfs = Parallel(n_jobs=n_jobs, verbose=10)(delayed(aggregate)(dataset[idx]) for idx in range(sample_size))
    summary_dfs = [df.set_index("Name") for df in summary_dfs if df is not None]
    summary_df = pd.concat(summary_dfs, axis=1).reset_index()
    print(f"Computed aggregation strategy summary: {time.time() - start} s")

    # Compute the correlation matrices: Pearson, Spearman, Kendall
    start = time.time()
    correlations = compute_correlations(summary_df)
    print(f"Computed correlation matrices: {time.time() - start} s")

    if dataset_name is None:
        try:
            dataset_name = dataset.get_info()['dataset_name']
        except:
            dataset_name = "" # NOTE: Please add dataset_name member to dataset class

    # Add dataset column to summary_df and save to csv
    summary_df = summary_df.T
    summary_df = summary_df.reset_index()
    summary_df.columns = summary_df.iloc[0]           # Set first row as header
    summary_df = summary_df.drop(index=0).reset_index(drop=True)  # Drop old header row
    summary_df.insert(loc=1, column="dataset_name", value=dataset_name)
    summary_df.rename(columns={'Name': 'uq_map_name'}, inplace=True)
    out_name = f"aggregation_value_summary_{dataset_name}"
    summary_df.to_csv(os.path.join("output", "tables", f"{out_name }.csv"), index=False)
    print(f"Aggregation value summary {out_name}.csv saved to output folder.")
        
    for correlation_type, corr_matrix in correlations.items():
        out_name = f"correlation_matrix_{correlation_type}_{dataset_name}"

        # Save to csv
        corr_matrix.to_csv(os.path.join("output", "tables", f"{out_name }.csv"))
        print(f"Correlation matrix {out_name}.csv saved to output folder.")

        # Save heatmap as png
        save_correlation_matrix_plot(corr_matrix, out_name, os.path.join("output", "figures"))
        print(f"Correlation heatmap {out_name}.png saved to output folder.")




import argparse

from datasets.ADE20K.ade20k_loader import ADE20K
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
            model_id = "resnest_s101-d8_fcn_4xb4-160k_ade20k-512x512" if model_name == "resnest" else "deeplabv3_r50-d8_4xb4-160k_ade20k-512x512"

            image_dir = '/fast/AG_Kainmueller/data/ADEChallengeData2016/images/validation'
            label_dir = '/fast/AG_Kainmueller/data/ADEChallengeData2016/annotations/validation'
            prediction_dir = f'/fast/AG_Kainmueller/data/ADEChallengeData2016/predictions/{model_id}/predictions/'
            uq_map_dir = f'/fast/AG_Kainmueller/data/UQ_maps/ADE20K/validation_{model_name}/semantic/{args.uq_method}/pu/'
            metadata_dir = '/fast/AG_Kainmueller/data/ADEChallengeData2016/objectInfo150.json'
            config_file = f'evaluation/configs/ade20k_{model_name}.yaml' # or also 'evaluation/configs/ade20k_resnest.yaml'
            config = load_dataset_config(config_file) 
            dataset = ADE20K(config['image_dir'],
                            config['label_dir'],
                            config['uq_map_dir'],
                            config['prediction_dir'],
                            config['metadata_dir'])
            dataset.num_classes = 150
            evaluate_correlation(dataset, args.sample_size, args.num_workers)
    

    if DATASET == "arctique":
        for task in ['instance', 'semantic']:
            noise_levels = ['0_00', '1_00']
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
                evaluate_correlation(dataset, args.sample_size, args.num_workers, dataset_name)


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
        evaluate_correlation(dataset, args.sample_size, args.num_workers, dataset_name)



    if DATASET == "lidc":
        if (am.patch_aggregation, 200) in focus_strategy_list:
            print(f"NOTE: We remove patch aggregation with patch size 200 because uq maps are smaller than 200x200.")
            focus_strategy_list.pop(focus_strategy_list.index((am.patch_aggregation, 200)))

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
                evaluate_correlation(dataset, args.sample_size, args.num_workers, dataset_name)


    if DATASET == "lizard":
        if (am.patch_aggregation, 200) in focus_strategy_list:
            print(f"NOTE: We remove patch aggregation with patch size 200 because uq maps are smaller than 200x200.")
            focus_strategy_list.pop(focus_strategy_list.index((am.patch_aggregation, 200)))

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
        evaluate_correlation(dataset, args.sample_size, args.num_workers, dataset_name)


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
        dataset_name = f"cityscapes_{args.uq_method}_pu"
        dataset.num_classes = 32
        evaluate_correlation(dataset, args.sample_size, args.num_workers, dataset_name)

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
        dataset_name = f"gta_{args.uq_method}_pu"
        dataset.num_classes = 32
        evaluate_correlation(dataset, args.sample_size, args.num_workers, dataset_name)