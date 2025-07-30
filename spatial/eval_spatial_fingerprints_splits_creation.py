import argparse
import os
import pandas as pd
import toml
import json

def load_config(path: str) -> dict:
    """Loads a TOML configuration file."""
    print(f"Loading configuration from: {path}")
    with open(path, 'r') as f:
        return toml.load(f)

def run_analysis_pipeline(paths: dict):
    """
    Finds the intersection between spatial data files and test split files,
    then stores the new subsets in the output directory, preserving the index.
    """
    output_dir = '/fast/AG_Kainmueller/vguarin/aggrigator_experiments/output/tables/spatial_fingerprints_eval_subsets/'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # --- Process In-Distribution (ID) data ---
        id_spatial_path = paths.get('id_spatial')
        id_test_split_path = paths.get('id_test_split')

        if not (id_spatial_path and os.path.exists(id_spatial_path)):
            print(f"Info: ID spatial data path not found or not provided ('{id_spatial_path}'). Skipping ID processing.")
        else:
            print(f"--- Processing ID file: {os.path.basename(id_spatial_path)} ---")
            print(f"Reading ID spatial data from: {id_spatial_path}")
            # Use the first column of the CSV as the DataFrame index
            id_spatial_df = pd.read_csv(id_spatial_path, index_col=0)
            
            if 'gta' in id_spatial_path:
                id_spatial_df.index = [f"{int(i):05d}" for i in id_spatial_df.index]

            if id_test_split_path and os.path.exists(id_test_split_path):
                print(f"Reading ID test split from: {id_test_split_path}")
                id_keys = []
                if id_test_split_path.endswith('.json'):
                    with open(id_test_split_path, 'r') as f:
                        id_keys = json.load(f)
                elif id_test_split_path.endswith('.csv'):
                    id_test_samples_df = pd.read_csv(id_test_split_path)
                    id_keys = id_test_samples_df.iloc[:, 0].unique().tolist()
                else:
                    print(f"Warning: Unrecognized split file format for '{id_test_split_path}'. Must be .json or .csv.")

                if id_keys:
                    # Filter the DataFrame based on its index
                    id_subset_df = id_spatial_df[id_spatial_df.index.isin(id_keys)]
                    
                    id_output_filename = os.path.basename(id_spatial_path)
                    id_output_path = os.path.join(output_dir, id_output_filename)
                    
                    print(f"Saving ID subset with {len(id_subset_df)} entries to: {id_output_path}")
                    # Save the DataFrame with its index
                    id_subset_df.to_csv(id_output_path, index=True)
                else:
                    print("Warning: Could not extract keys from ID split file. No subset created.")
            else:
                print(f"Warning: ID test split not found at '{id_test_split_path}'. No ID subset will be created.")

        # --- Process Out-of-Distribution (OOD) data ---
        ood_spatial_path = paths.get('ood_spatial')
        ood_test_split_path = paths.get('ood_test_split')

        if not (ood_spatial_path and os.path.exists(ood_spatial_path)):
            print(f"Info: OOD spatial data path not found or not provided ('{ood_spatial_path}'). Skipping OOD processing.")
            return

        print(f"--- Processing OOD file: {os.path.basename(ood_spatial_path)} ---")
        print(f"Reading OOD spatial data from: {ood_spatial_path}")
        # Use the first column of the CSV as the DataFrame index
        ood_spatial_df = pd.read_csv(ood_spatial_path, index_col=0)
        
        ood_subset_df = None
        ood_output_filename = os.path.basename(ood_spatial_path)
        ood_output_path = os.path.join(output_dir, ood_output_filename)

        if ood_test_split_path and os.path.exists(ood_test_split_path):
            print(f"Reading OOD test split from: {ood_test_split_path}")
            ood_keys = []
            if ood_test_split_path.endswith('.json'):
                with open(ood_test_split_path, 'r') as f:
                    ood_keys = json.load(f)
            elif ood_test_split_path.endswith('.csv'):
                ood_test_samples_df = pd.read_csv(ood_test_split_path)
                ood_keys = ood_test_samples_df.iloc[:, 0].unique().tolist()
            
            if ood_keys:
                 # Filter the DataFrame based on its index
                 ood_subset_df = ood_spatial_df[ood_spatial_df.index.isin(ood_keys)]
            else:
                print("Warning: Could not extract keys from OOD split file. Using full OOD dataset instead.")
                ood_subset_df = ood_spatial_df
        else:
            print("OOD test split not found or path not provided. Using the full OOD spatial file as the subset.")
            ood_subset_df = ood_spatial_df
        
        print(f"Saving OOD subset with {len(ood_subset_df)} entries to: {ood_output_path}")
        # Save the DataFrame with its index
        ood_subset_df.to_csv(ood_output_path, index=True)

    except FileNotFoundError as e:
        print(f"Error: A file was not found during processing. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    print(f"--- Analysis complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates subsets of spatial fingerprints split by samples used in evaluation for point cloud visualization")
    parser.add_argument('--config', type=str, default='/fast/AG_Kainmueller/vguarin/aggrigator_experiments/spatial/spatial_configs/arctique.toml', help='Path to config TOML file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    dataset_name = config['dataset']['dataset_name']
    
    print(f"Loaded configuration for dataset: '{dataset_name}'")

    analysis_paths_templates = {
        'id_spatial': config['paths'].get('id_csv_path_spatial'),
        'ood_spatial': config['paths'].get('ood_csv_path_spatial'),
        'id_test_split': config['splits'].get('id_json_path_spatial'),
        'ood_test_split': config['splits'].get('ood_json_path_spatial')
    }
    
    main_loop_executed = False

    if dataset_name.startswith('lidc'):
        main_loop_executed = True
        task = config['dataset']['task']
        original_variation = config['dataset']['variation']
        for var in ['malignancy', 'texture']:
            print(f"\n\n{'='*25} PROCESSING VARIATION: {var.upper()} {'='*25}")
            current_paths = {k: v.replace(original_variation, var) for k, v in analysis_paths_templates.items() if v}
            run_analysis_pipeline(current_paths)
            
    if dataset_name.startswith('arctique'):
        main_loop_executed = True
        original_task = config['dataset']['task']
        original_variation = config['dataset']['variation']
        original_noise = config['dataset']['noise_level']
        for task, var, ns in zip(['semantic', 'instance'], ['blood_cells', 'nuclei_intensity'], ['0_75', '0_50']):
            print(f"\n\n{'='*25} PROCESSING TASK: {task.upper()}; VARIATION: {var.upper()}  {'='*25}")
            current_paths = {}
            for k, v_template in analysis_paths_templates.items():
                if not v_template:
                    current_paths[k] = None
                    continue
                
                temp_path = v_template.replace(original_task, task).replace(original_variation, var)
                if k.startswith('ood'):
                    temp_path = temp_path.replace(original_noise, ns)
                current_paths[k] = temp_path
            
            print("Generated paths for this run:", current_paths)
            run_analysis_pipeline(current_paths)
            
    if dataset_name.startswith('lizard'):
        main_loop_executed = True
        original_task = config['dataset']['task']
        original_variation = config['dataset']['variation']
        for task in ['semantic', 'instance']:
            print(f"\n\n{'='*25} PROCESSING TASK: {task.upper()} {'='*25}")
            current_paths = {k: v.replace(original_task, task) for k, v in analysis_paths_templates.items() if v}
            run_analysis_pipeline(current_paths)
    
    if dataset_name.startswith('wormbodies'):
        main_loop_executed = True
        original_task = config['dataset']['task']
        original_variation = config['dataset']['variation']
        for var in ['nematodes', 'protists']:
            print(f"\n\n{'='*25} PROCESSING VARIATION: {var.upper()} {'='*25}")
            current_paths = {}
            for k, v_template in analysis_paths_templates.items():
                if not v_template:
                    current_paths[k] = None
                    continue
                
                if k.startswith('id') and k.endswith('spatial'):
                    current_paths[k] = v_template
                else:
                    current_paths[k] = v_template.replace(original_variation, var)

            print("Generated paths for this run:", current_paths)
            run_analysis_pipeline(current_paths)

    if not main_loop_executed:
        print("\nDataset name in config did not match any known processing loops.")
        print("Running a default analysis based on the provided config without loops.")
        run_analysis_pipeline(analysis_paths_templates)

    print(f"\n{'='*25} SUBSETS CREATION COMPLETE {'='*25}")