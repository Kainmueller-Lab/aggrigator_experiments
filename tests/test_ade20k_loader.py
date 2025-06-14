import unittest

from datasets.datalaoders.ade20k_loader import ADE20K


class TestADE20KDataset(unittest.TestCase):

    def setUp(self):
        self.image_path = '/fast/AG_Kainmueller/data/ADEChallengeData2016/images/validation/'
        self.mask_path = '/fast/AG_Kainmueller/data/ADEChallengeData2016/annotations/validation/'
        self.uq_map_path = '/fast/AG_Kainmueller/data/UQ_maps/ADE20K/validation_deeplabv3/semantic/dropout/pu/'
        self.pred_path = '/fast/AG_Kainmueller/data/ADEChallengeData2016/predictions/deeplabv3_r50-d8_4xb4-160k_ade20k-512x512/predictions/'
        self.semantic_mapping_path = '/fast/AG_Kainmueller/data/ADEChallengeData2016/objectInfo150.json'


    def test_dataset_loading(self):
        dataset = ADE20K(self.image_path,
                         self.mask_path,
                         self.uq_map_path,
                         self.pred_path,
                         self.semantic_mapping_path)

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
