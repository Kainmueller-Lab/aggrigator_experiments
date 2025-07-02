import os
import yaml
import unittest
import numpy as np
import pandas as pd


from datasets.ADE20K.ade20k_loader import ADE20K
from aggrigator.uncertainty_maps import UncertaintyMap
from aggrigator.spatial_decomposition import spatial_decomposition # NOTE: This is only available on the develop branch of the aggrigator repo. Use "pip install -e ." to install the package.




class TestCorrelation(unittest.TestCase):
    def test_ade20k_fingerprint(self):
        with open(os.path.join(os.path.dirname(__file__), '..', 'evaluation', 'configs', 'ade20k_deeplabv3.yaml'), 'r') as f:
            config = yaml.safe_load(f)

        dataset = ADE20K(config['image_dir'],
                         config['label_dir'],
                         config['uq_map_dir'],
                         config['prediction_dir'],
                         config['metadata_dir'])
        dataset_info = dataset.get_info()

        self.assertEqual(len(dataset), 2000)

        sample_size = 1


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
            if dataset_info['num_classes'] is not None:
                uq_array = uq_array / np.log(dataset_info['num_classes'])

            # Compute spatial decomposition for all spatial measures
            spatial_measures = ["moran"]
            window_size = 3
            uq_map = UncertaintyMap(array=uq_array, mask=None, name=sample_name)
            measure_mass_ratios = {measure: spatial_decomposition(uq_map, window_size=window_size, spatial_measure=measure)[3] for measure in spatial_measures}
            return (sample_name, measure_mass_ratios)

        # Decompose all UQ maps
        measure_mass_ratios = [get_measure_mass_ratios(dataset[idx]) for idx in range(sample_size)]
        measure_mass_ratio_df = pd.DataFrame.from_dict(dict(measure_mass_ratios), orient='index')
        moran_mass_ratio = measure_mass_ratio_df.values[0][0]
        expected = 0.48384669
        self.assertAlmostEqual(moran_mass_ratio, expected, places=4)





if __name__ == "__main__":
    unittest.main()
