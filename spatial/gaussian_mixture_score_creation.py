import pandas as pd
import numpy as np
import toml
import os
import matplotlib.pyplot as plt
import argparse
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, auc
import json

def load_config(config_path: str = "config.toml"):
    """Loads configuration from a TOML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")
    return toml.load(config_path)

def load_fingerprints_from_csv(filepath: str) -> pd.DataFrame:
    """Loads spatial fingerprints from a CSV file."""
    try:
        df = pd.read_csv(filepath, index_col=0)
        return df[['moran', 'entropy', 'eds']]
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return None

def get_or_create_split(id_full_df: pd.DataFrame, split_dir: str, base_filename: str):
    """
    Checks for existing train/test split JSON files. If they don't exist,
    creates a 50/50 split and saves it.

    For the 'gta' dataset, it saves correctly formatted, zero-padded 6-digit
    string indices. For all other datasets, it saves the raw integer indices.
    """
    train_split_path = os.path.join(split_dir, f"{base_filename}_train_split.json")
    test_split_path = os.path.join(split_dir, f"{base_filename}_test_split.json")

    if os.path.exists(train_split_path) and os.path.exists(test_split_path):
        print(f"Found existing split files. Loading from:\n  - {train_split_path}\n  - {test_split_path}")
        with open(train_split_path, 'r') as f:
            train_indices_loaded = json.load(f)
        with open(test_split_path, 'r') as f:
            test_indices_loaded = json.load(f)
            
        # For DataFrame slicing, we need the original integer indices.
        # This robustly converts loaded indices (whether string or int) back to int.
        train_indices = [int(i) for i in train_indices_loaded]
        test_indices = [int(i) for i in test_indices_loaded]

    else:
        print("No existing split found. Creating a new 50/50 split.")
        # These are the original, unformatted indices (e.g., integers)
        indices = id_full_df.index.to_list()
        train_indices, test_indices = train_test_split(indices, test_size=0.5, random_state=42)

        # --- MODIFICATION: Conditional Formatting Logic ---
        # Check if the dataset is 'gta' based on the filename structure.
        if 'gta' in base_filename:
            print("GTA dataset detected. Applying 5-digit zero-padding to indices for saving.")
            # Format the numeric indices into 5-digit zero-padded strings.
            train_indices_to_save = [f"{int(i):05d}" for i in train_indices]
            test_indices_to_save = [f"{int(i):05d}" for i in test_indices]
            print(f"Example formatted index: {train_indices[0]} -> {train_indices_to_save[0]}")
        else:
            # For all other datasets, save the original integer indices.
            print(f"Non-GTA dataset ('{base_filename}') detected. Saving raw integer indices.")
            train_indices_to_save = train_indices
            test_indices_to_save = test_indices
        # --- END MODIFICATION ---

        # Save the appropriate lists (either formatted or raw) to the JSON files.
        with open(train_split_path, 'w') as f:
            json.dump(train_indices_to_save, f, indent=4)
        with open(test_split_path, 'w') as f:
            json.dump(test_indices_to_save, f, indent=4)
        print(f"Saved new splits to:\n  - {train_split_path}\n  - {test_split_path}")

    # Use the ORIGINAL, unformatted integer indices to locate data in the DataFrame.
    id_train_df = id_full_df.loc[train_indices]
    id_test_df = id_full_df.loc[test_indices]
    
    return id_train_df, id_test_df

def find_best_gmm(id_fingerprints: pd.DataFrame, max_components: int = 8):
    """Tests different numbers of GMM components and finds the best one using BIC."""
    id_data = id_fingerprints.to_numpy()
    bic_scores = []
    component_range = range(1, max_components + 1)
    print(f"\n--- Model Selection on Training Data: Finding best GMM components ---")
    for n in component_range:
        gmm = GaussianMixture(n_components=n, random_state=42, n_init=10)
        gmm.fit(id_data)
        bic_scores.append(gmm.bic(id_data))
    best_n_components = np.argmin(bic_scores) + 1
    print(f"CONCLUSION: Optimal number of components is {best_n_components} (lowest BIC).")
    return best_n_components

def build_id_gmm_model(id_train_fingerprints: pd.DataFrame, n_components: int):
    """Builds a GMM from the ID training fingerprints."""
    id_train_data = id_train_fingerprints.to_numpy()
    print("\n--- Building final GMM Model on ID-Train data ---")
    gmm_model = GaussianMixture(n_components=n_components, random_state=42, n_init=10)
    gmm_model.fit(id_train_data)
    id_log_likelihoods = gmm_model.score_samples(id_train_data)
    max_density = np.exp(np.max(id_log_likelihoods))
    print(f"Using {n_components} component(s). Max likelihood density on training data: {max_density:.4f}\n")
    return gmm_model, max_density

def calculate_scores(fingerprints: pd.DataFrame, id_model):
    """MODIFIED: Calculates only the robust NLL-based scores."""
    gmm_model, max_density = id_model
    data = fingerprints.to_numpy()
    
    log_p_values = gmm_model.score_samples(data)
    ood_scores_nll = -log_p_values
    
    nll_min = -np.log(max_density + 1e-20)
    ood_scores_nll_zero_floored = ood_scores_nll - nll_min
    
    results_df = fingerprints.copy()
    results_df['ood_score_nll'] = ood_scores_nll
    results_df['ood_score_nll_zero_floored'] = ood_scores_nll_zero_floored
    
    return results_df

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/fast/AG_Kainmueller/vguarin/aggrigator_experiments/spatial/spatial_configs/arctique.toml', help='Path to config TOML file')
    return parser.parse_args()

def run_analysis_pipeline(id_csv_path: str, ood_csv_path: str, base_filename: str):
    """Loads data, handles splits, and runs the full GMM OoD analysis."""
    print(f"ID Path: {id_csv_path}")
    print(f"OoD Path: {ood_csv_path}\n")

    id_full_df = load_fingerprints_from_csv(id_csv_path)
    if id_full_df is None:
        print("Skipping analysis due to missing ID data.")
        return
    
    save_dir = os.path.join(os.getcwd(), 'spatial', 'splits')
    os.makedirs(save_dir, exist_ok=True)

    id_train_df, id_test_df = get_or_create_split(id_full_df, save_dir, base_filename)
    best_k = find_best_gmm(id_train_df, max_components=8)
    id_model = build_id_gmm_model(id_train_df, n_components=best_k)

    print("--- Calculating scores for ID-Test data ---")
    id_test_results = calculate_scores(id_test_df, id_model)
    id_test_results['is_ood'] = 0

    ood_fingerprints = load_fingerprints_from_csv(ood_csv_path)
    if ood_fingerprints is not None:
        print("\n--- Calculating scores for OoD data ---")
        ood_results = calculate_scores(ood_fingerprints, id_model)
        ood_results['is_ood'] = 1

        final_results = pd.concat([id_test_results, ood_results])
        
        max_nll = final_results['ood_score_nll_zero_floored'].max()
        print(f"\nDynamically scaling scores based on observed max NLL: {max_nll:.4f}")
        final_results['ood_score_normalized'] = (final_results['ood_score_nll_zero_floored'] / (max_nll + 1e-9)).clip(0, 1)

        cols = ['moran', 'entropy', 'eds', 'ood_score_normalized', 'ood_score_nll', 'ood_score_nll_zero_floored', 'is_ood']
        final_results = final_results[cols]

        res_dir = os.path.join(os.getcwd(), 'spatial', 'results')
        os.makedirs(res_dir, exist_ok=True)
        results_filename = os.path.join(res_dir, f"{base_filename}_scores.csv")
        final_results.to_csv(results_filename)
        print(f"\nSaved combined scores to:\n  - {results_filename}")
        
        # --- ENHANCED SUMMARY SECTION ---
        print("\n--- Final Summary ---")

        # --- NEW: AUROC & ROC Curve Analysis ---
        y_true = final_results['is_ood']
        y_score = final_results['ood_score_normalized']
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auroc_score = auc(fpr, tpr)
        # auroc_score = roc_auc_score(y_true, y_score)
        
        print("\n--- ROC / AUROC Performance Analysis ---")
        print(f"Area Under the ROC Curve (AUROC): {auroc_score:.4f}")

        # Generate and save ROC curve plot
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUROC = {auroc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(f'Receiver Operating Characteristic: {base_filename}')
        plt.legend(loc="lower right")
        
        roc_plot_filename = os.path.join(res_dir, f"{base_filename}_roc_curve.png")
        plt.savefig(roc_plot_filename)
        plt.close() # Close the plot to free memory
        print(f"Saved ROC curve plot to:\n  - {roc_plot_filename}")
        
        print("\n--- Misclassification Examples ---")
        
        print("\nTop 5 ID samples that look most like OoD (Highest Score):")
        print(final_results[final_results['is_ood'] == 0].sort_values(by='ood_score_normalized', ascending=False).head())

        print("\nTop 5 OoD samples that look most like ID (Lowest Score):")
        print(final_results[final_results['is_ood'] == 1].sort_values(by='ood_score_normalized', ascending=True).head())
        
    else:
        print("Skipping OoD scoring due to missing OoD data.")
        # Handle case with only ID data
        res_dir = os.path.join(os.getcwd(), 'spatial', 'results')
        os.makedirs(res_dir, exist_ok=True)
        results_filename = os.path.join(res_dir, f"{base_filename}_scores.csv")
        max_nll_id = id_test_results['ood_score_nll_zero_floored'].max()
        id_test_results['ood_score_normalized'] = (id_test_results['ood_score_nll_zero_floored'] / (max_nll_id + 1e-9)).clip(0, 1)
        id_test_results.to_csv(results_filename)
        print(f"\nSaved scores for ID-Test data to:\n  - {results_filename}")


if __name__ == "__main__":
    # The main execution block remains the same
    args = parse_args()
    config = load_config(args.config)
    dataset_name = config['dataset']['dataset_name']
    
    os.makedirs(os.path.join(os.getcwd(), 'spatial', 'splits'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), 'spatial', 'results'), exist_ok=True)
    
    print(f"Loaded configuration for dataset: '{dataset_name}'")

    if dataset_name.startswith('arctique'):
        id_path_template = config['paths']['id_csv_path']
        ood_path_template = config['paths']['ood_csv_path']
        original_task = config['dataset']['task']
        variation = config['dataset']['variation']
        uq_method = config['dataset']['variation']
        for task in ['semantic']: #, 'instance'
            print(f"\n\n{'='*25} PROCESSING TASK: {task.upper()} {'='*25}")
            current_id_path = id_path_template.replace(original_task, task)
            current_ood_path = ood_path_template.replace(original_task, task)
            base_filename = f"{task}_{dataset_name}_{variation}_pu"
            run_analysis_pipeline(current_id_path, current_ood_path, base_filename)
    
    elif dataset_name.startswith('lidc'):
        id_path_template = config['paths']['id_csv_path']
        ood_path_template = config['paths']['ood_csv_path']
        task = config['dataset']['task']
        original_variation = config['dataset']['variation']
        for var in ['malignancy', 'texture']:
            print(f"\n\n{'='*25} PROCESSING VARIATION: {var.upper()} {'='*25}")
            current_id_path = id_path_template.replace(original_variation, var)
            current_ood_path = ood_path_template.replace(original_variation, var)
            base_filename = f"{task}_{dataset_name}_{var}_pu"
            run_analysis_pipeline(current_id_path, current_ood_path, base_filename)
            
    else:
        print("Standard dataset detected. Running a single analysis.")
        id_csv_path = config['paths']['id_csv_path']
        ood_csv_path = config['paths']['ood_csv_path'] 
        task = config['dataset']['task']
        variation = config['dataset']['variation']
        base_filename = f"{task}_{dataset_name}_{variation}_pu" if variation != "" else f"{task}_{dataset_name}_pu" 
        run_analysis_pipeline(id_csv_path, ood_csv_path, base_filename)

    print(f"\n{'='*25} ANALYSIS COMPLETE {'='*25}")