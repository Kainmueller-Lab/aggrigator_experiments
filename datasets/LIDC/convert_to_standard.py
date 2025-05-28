import os
import numpy as np
import nibabel as nib
from pathlib import Path
import glob
import json

def load_and_sum_nifti(file_path):
    """Load a NIfTI file and sum over the last dimension (z-axis)"""
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        # Sum over the last dimension (z-axis)
        summed_data = np.sum(data, axis=-1) #TODO: to confirm with Carsten 
        return summed_data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def process_pred_entropy_folder(pred_entropy_path):
    """Process all .nii.gz files in a pred_entropy folder"""
    nii_files = glob.glob(os.path.join(pred_entropy_path, "*.nii.gz"))
    
    if not nii_files:
        print(f"No .nii.gz files found in {pred_entropy_path}")
        return None, None
    
    processed_data = []
    sample_indices = []
    
    for nii_file in sorted(nii_files):
        print(f"Processing: {os.path.basename(nii_file)}")
        summed_data = load_and_sum_nifti(nii_file)
        
        if summed_data is not None:
            processed_data.append(summed_data)
            # Extract sample index (e.g., "0014_01" from "0014_01.nii.gz")
            filename = os.path.basename(nii_file)
            sample_idx = filename.replace('.nii.gz', '')
            sample_indices.append(sample_idx)
        else:
            print(f"Skipping {nii_file} due to loading error")
    
    if processed_data:
        # Stack all samples into n_samples x H x W array
        result = np.stack(processed_data, axis=0)
        print(f"Created array with shape: {result.shape}")
        return result, sample_indices
    else:
        print("No data was successfully processed")
        return None, None

def create_metadata(method, method_name, ood_var, data_noise, data_type, sample_indices):
    """Create metadata dictionary for JSON file"""
    metadata = {
        "root_dir": f"/fast/AG_Kainmueller/data/ValUES/FirstCycle/{method}/test_results/{ood_var}_fold0_seed123/",
        "dataset": "lidc-idri",
        "data_modality": "id" if data_noise == "0_00" else "ood",
        "image_noise": data_noise,
        "mask_noise": "0",
        "model": "uncertainty_modeling.models.unet3D_module.UNet3D",
        "uq_method": method_name,
        "exp": f"/fast/AG_Kainmueller/data/ValUES/FirstCycle/{method}/{ood_var}_fold0_seed123/checkpoints/",
        "sample_idx": sample_indices,
        "calibr": False
    }
    return metadata

def main():
    # Base directory
    base_dir = "/fast/AG_Kainmueller/data/ValUES/"
    cycle = "FirstCycle/"
    
    # Create UQ_maps and UQ_metadata folders if they don't exist
    uq_maps_dir = os.path.join(base_dir, "UQ_maps")
    uq_metadata_dir = os.path.join(base_dir, "UQ_metadata")
    os.makedirs(uq_maps_dir, exist_ok=True)
    os.makedirs(uq_metadata_dir, exist_ok=True)
    print(f"UQ_maps directory: {uq_maps_dir}")
    print(f"UQ_metadata directory: {uq_metadata_dir}")
    
    # Methods and their corresponding folder names
    methods = ["Dropout", "Ensemble", "Softmax", "TTA"]
    method_names = ["dropout", "ensemble", "softmax", "tta"]  # lowercase for filename
    
    # OOD variations
    ood_variations = ["malignancy", "texture"]
    
    # Data noise types
    data_noise_mapping = {"id": "0_00", "ood": "1_00"}
    
    for method, method_name in zip(methods, method_names):
        method_path = os.path.join(base_dir, cycle, method, "test_results")
        
        if not os.path.exists(method_path):
            print(f"Method path does not exist: {method_path}")
            continue
            
        print(f"\nProcessing method: {method}")
        
        for ood_var in ood_variations:
            # Look for the specific folder pattern
            folder_pattern = f"{ood_var}_fold0_seed123"
            folder_path = os.path.join(method_path, folder_pattern)
            
            if not os.path.exists(folder_path):
                print(f"Folder does not exist: {folder_path}")
                continue
                
            print(f"  Processing OOD variation: {ood_var}")
            
            for data_type, data_noise in data_noise_mapping.items():
                data_type_path = os.path.join(folder_path, data_type)
                
                if not os.path.exists(data_type_path):
                    print(f"    Data type path does not exist: {data_type_path}")
                    continue
                    
                pred_entropy_path = os.path.join(data_type_path, "pred_entropy")
                
                if not os.path.exists(pred_entropy_path):
                    print(f"    pred_entropy path does not exist: {pred_entropy_path}")
                    continue
                
                print(f"    Processing {data_type} data...")
                
                # Process the pred_entropy folder
                processed_array, sample_indices = process_pred_entropy_folder(pred_entropy_path)
                
                if processed_array is not None:
                    # Generate filename: fgbg_noise_0_{ood_variation}_{data_noise}_{method}_pu.npy
                    base_filename = f"fgbg_noise_0_{ood_var}_{data_noise}_{method_name}_pu"
                    npy_filename = f"{base_filename}.npy"
                    json_filename = f"{base_filename}.json"
                    
                    npy_output_path = os.path.join(uq_maps_dir, npy_filename)
                    json_output_path = os.path.join(uq_metadata_dir, json_filename)
                    
                    # Save the numpy array
                    np.save(npy_output_path, processed_array)
                    print(f"    Saved: {npy_filename} with shape {processed_array.shape}")
                    
                    # Create and save metadata
                    metadata = create_metadata(method, method_name, ood_var, data_noise, data_type, sample_indices)
                    with open(json_output_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    print(f"    Saved: {json_filename}")
                else:
                    print(f"    Failed to process {pred_entropy_path}")

if __name__ == "__main__":
    print("Starting NIfTI to NumPy conversion...")
    main()
    print("Conversion completed!")