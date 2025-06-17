import os
import yaml
import unittest
import numpy as np
import pandas as pd


from evaluation.scripts.evaluate_correlation import compute_correlations
from datasets.ADE20K.ade20k_loader import ADE20K
from aggrigator.uncertainty_maps import UncertaintyMap
from aggrigator.methods import AggregationMethods as am
from aggrigator.summary import AggregationSummary


class TestCorrelation(unittest.TestCase):
    def setUp(self):
        self.focus_strategy_list = [
            (am.above_threshold_mean, 0.3),
            (am.above_threshold_mean, 0.5),
            (am.above_threshold_mean, 0.7),
            (am.above_threshold_mean, 0.9),
            (am.above_threshold_mean, 0.95),
        ]


    def test_ade20k_correlation(self):
        with open(os.path.join(os.path.dirname(__file__), '..', 'configs', 'ade20k_deeplabv3.yaml'), 'r') as f:
            config = yaml.safe_load(f)

        dataset = ADE20K(config['image_dir'],
                         config['label_dir'],
                         config['uq_map_dir'],
                         config['prediction_dir'],
                         config['metadata_dir'])
        dataset_info = dataset.get_info()

        self.assertEqual(len(dataset), 2000)

        sample_size = 10

        def aggregate(sample):
            # Load uncertainty maps and masks from dataset
            mask = sample['mask']
            uq_array = sample['uq_map']

            # Slice if 3D
            if uq_array.ndim == 3:
                print(f"Warning: 3D UQ map detected. Only 2D slices are used for correlation matrix.")
                mid_slice = uq_array.shape[0] // 2
                uq_array = uq_array[mid_slice, :, :]
                mask = mask[mid_slice, :, :]

            # Ignore too small images bc of patch aggregation with patch size 200
            h, w = uq_array.shape
            if h < 200 or w < 200:
                print(f"Warning: UQ map {sample['sample_name']} is too small for patch aggregation with patch size 200.")
                return None
            
            # Replace negative values with zero
            # NOTE: Such values (close to zero) sometimes occur and need to be dealt with.
            uq_array = np.where(uq_array < 0, 0, uq_array)
            
            # Normalize arrays by ln(K) where K is number of classes if UQ maps are not normalized in dataloader
            if dataset_info['num_classes'] is not None:
                uq_array = uq_array / np.log(dataset_info['num_classes'])

            # Apply aggregation strategies
            uq_map = UncertaintyMap(array=uq_array, mask=mask, name=sample['sample_name'])
            summary = AggregationSummary(self.focus_strategy_list, num_cpus=1)
            return summary.apply_methods([uq_map], save_to_excel=False, do_plot=False, max_value=1.0)
        
        # Aggregate all UQ maps
        summary_dfs = [aggregate(dataset[idx]) for idx in range(sample_size)]
        summary_dfs = [df.set_index("Name") for df in summary_dfs if df is not None]
        summary_df = pd.concat(summary_dfs, axis=1).reset_index()

        # Check that all above threshold aggregations are zero for ADE20K for a high threshold
        above_thresh_df = summary_df[summary_df["Name"]=="above_threshold_mean_0.9"]
        values = above_thresh_df.values[1:]
        self.assertTrue(np.all(values == 0.0))

        # Leading to NaNs in the correlation matrix rows and columns (i.e. white columns in heatmap)
        correlations = compute_correlations(summary_df)
        corrs_for_tresh = correlations["pearson"]["above_threshold_mean_0.9"].values
        self.assertTrue(np.isnan(corrs_for_tresh).all())
    





if __name__ == "__main__":
    unittest.main()
