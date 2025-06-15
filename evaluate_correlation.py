import argparse

from evaluation.scripts.evaluate_correlation import evaluate_correlation, load_dataset_config
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
    
    evaluate_correlation(dataset, args.sample_size, args.num_workers)

    

