import os
import yaml
import unittest

from datasets.ADE20K.ade20k_loader import ADE20K


class TestADE20KDataset(unittest.TestCase):
    def test_deeplabv3(self):
        with open(os.path.join(os.path.dirname(__file__), '..', 'configs', 'ade20k_deeplabv3.yaml'), 'r') as f:
            config = yaml.safe_load(f)

        dataset = ADE20K(config['image_dir'],
                         config['label_dir'],
                         config['uq_map_dir'],
                         config['prediction_dir'],
                         config['metadata_dir'])

        self.assertEqual(len(dataset), 2000)

        sample = dataset[0]
        self.assertIn('image', sample)
        self.assertIn('mask', sample)
        self.assertIn('uq_map', sample)
        self.assertIn('prediction', sample)
        self.assertIn('sample_name', sample)

        self.assertEqual(sample['image'].shape, (3, 512, 683))
        self.assertEqual(sample['mask'].shape, (512, 683))
        self.assertEqual(sample['uq_map'].shape, (512, 683))
        self.assertEqual(sample['prediction'].shape, (512, 683))

        self.assertIsInstance(dataset.get_semantic_mapping(), dict)
        self.assertEqual(len(dataset.get_semantic_mapping()), 150)
        self.assertIsInstance(dataset.get_info(), dict)


    def test_resnest(self):
        with open(os.path.join(os.path.dirname(__file__), '..', 'configs', 'ade20k_resnest.yaml'), 'r') as f:
            config = yaml.safe_load(f)

        dataset = ADE20K(config['image_dir'],
                         config['label_dir'],
                         config['uq_map_dir'],
                         config['prediction_dir'],
                         config['metadata_dir'])

        self.assertEqual(len(dataset), 2000)

        sample = dataset[0]
        self.assertIn('image', sample)
        self.assertIn('mask', sample)
        self.assertIn('uq_map', sample)
        self.assertIn('prediction', sample)
        self.assertIn('sample_name', sample)

        self.assertEqual(sample['image'].shape, (3, 512, 683))
        self.assertEqual(sample['mask'].shape, (512, 683))
        self.assertEqual(sample['uq_map'].shape, (512, 683))
        self.assertEqual(sample['prediction'].shape, (512, 683))

        self.assertIsInstance(dataset.get_semantic_mapping(), dict)
        self.assertEqual(len(dataset.get_semantic_mapping()), 150)
        self.assertIsInstance(dataset.get_info(), dict)


if __name__ == "__main__":
    unittest.main()
