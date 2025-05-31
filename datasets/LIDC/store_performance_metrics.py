import os
import json
from pathlib import Path

def extract_image_id_from_path(file_path):
    """Extract image ID from file path (e.g., '0057_06' from '/path/to/0057_06.npy')"""
    filename = os.path.basename(file_path)
    # Remove .npy extension
    return filename.replace('.npy', '')

def process_metrics_file(metrics_file_path, ood_var, data_noise, method_name, performance_metrics_dir):
    """Process metrics.json file and save individual metric files"""
    if not os.path.exists(metrics_file_path):
        print(f"    Metrics file does not exist: {metrics_file_path}")
        return False
    
    try:
        with open(metrics_file_path, 'r') as f:
            metrics_data = json.load(f)
        
        # Dictionary to store metrics by metric name
        metrics_by_type = {}
        
        # Process each file entry in the metrics
        for file_path, file_metrics in metrics_data.items():
            image_id = extract_image_id_from_path(file_path)
            
            # Process each metric for this image
            for metric_name, metric_value in file_metrics.items():
                # Clean metric name for filename (replace spaces with underscores)
                clean_metric_name = metric_name.replace(' ', '_')
                
                if clean_metric_name not in metrics_by_type:
                    metrics_by_type[clean_metric_name] = {}
                
                metrics_by_type[clean_metric_name][image_id] = metric_value
        
        # Save each metric type as a separate JSON file
        for metric_name, metric_dict in metrics_by_type.items():
            # Generate filename: fgbg_noise_0_{ood_var}_{data_noise}_{method_name}_{metric}.json
            metric_filename = f"fgbg_noise_0_{ood_var}_{data_noise}_{method_name}_{metric_name}.json"
            metric_output_path = os.path.join(performance_metrics_dir, metric_filename)
            
            # Save the metric dictionary
            with open(metric_output_path, 'w') as f:
                json.dump(metric_dict, f, indent=2)
            
            print(f"    Saved: {metric_filename} with {len(metric_dict)} samples")
        
        return True
            
    except Exception as e:
        print(f"    Error processing metrics file {metrics_file_path}: {e}")
        return False

def main():
    # Base directory
    base_dir = "/fast/AG_Kainmueller/data/ValUES/"
    cycle = "FirstCycle/"
    
    # Create Performance_metrics folder if it doesn't exist
    performance_metrics_dir = os.path.join(base_dir, "Performance_metrics")
    os.makedirs(performance_metrics_dir, exist_ok=True)
    
    print(f"Performance_metrics directory: {performance_metrics_dir}")
    
    # Methods and their corresponding folder names
    methods = ["Dropout", "Ensemble", "Softmax", "TTA"]
    method_names = ["dropout", "ensemble", "softmax", "tta"]  # lowercase for filename
    
    # OOD variations
    ood_variations = ["malignancy", "texture"]
    
    # Data noise types
    data_noise_mapping = {"id": "0_00", "ood": "1_00"}
    
    total_processed = 0
    total_failed = 0
    
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
                
                metrics_file_path = os.path.join(data_type_path, "metrics.json")
                
                print(f"    Processing {data_type} data...")
                
                # Process performance metrics
                success = process_metrics_file(metrics_file_path, ood_var, data_noise, method_name, performance_metrics_dir)
                
                if success:
                    total_processed += 1
                else:
                    total_failed += 1
    
    print(f"\n=== Processing Summary ===")
    print(f"Successfully processed: {total_processed} metrics files")
    print(f"Failed to process: {total_failed} metrics files")
    print(f"Output directory: {performance_metrics_dir}")

if __name__ == "__main__":
    print("Starting Performance Metrics Processing...")
    main()
    print("Processing completed!")