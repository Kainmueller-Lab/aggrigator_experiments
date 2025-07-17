import pandas as pd
import numpy as np
import toml
import os
import matplotlib.pyplot as plt
import argparse
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import json

def load_config(config_path: str = "config.toml"):
    """Loads configuration from a TOML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")
    return toml.load(config_path)

def load_spatial_fingerprints(filepath: str, base_filename: str) -> pd.DataFrame:
    """Loads spatial fingerprints, ensuring the index is treated as a string."""
    try:
        if 'arctique' in base_filename: 
            df = pd.read_csv(filepath, index_col=0)
        else:
            df = pd.read_csv(filepath, index_col=0, dtype={0: str})
        return df[['moran', 'entropy', 'eds']]
    except FileNotFoundError:
        print(f"Warning: The spatial data file '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the spatial CSV: {e}")
        return None

def load_magnitude_fingerprints(filepath: str, base_filename: str) -> pd.DataFrame:
    """
    Loads magnitude fingerprints, ensuring the index is treated as a string.
    """
    try:
        # FIX: Force the 'uq_map_name' column to be read as a string.
        if 'arctique' in base_filename:
             df = pd.read_csv(filepath, index_col='uq_map_name')
        else:
            df = pd.read_csv(filepath, index_col='uq_map_name', dtype={'uq_map_name': str})
        numeric_df = df.select_dtypes(include=np.number)
        feature_cols = [col for col in numeric_df.columns if not col.lower().startswith('gmm')]
        if not feature_cols:
            print(f"Warning: No numeric feature columns found in '{filepath}' after filtering.")
            return None
        return numeric_df[feature_cols]
    except FileNotFoundError:
        print(f"Warning: The magnitude data file '{filepath}' was not found.")
        return None
    except KeyError:
        print(f"Error: The required index column 'uq_map_name' was not found in '{filepath}'.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the magnitude CSV: {e}")
        return None

def get_or_create_split(id_full_df: pd.DataFrame, split_dir: str, base_filename: str):
    """
    Checks for existing train/test split JSON files. If they don't exist,
    creates a 50/50 split and saves it. Correctly handles string-based indices.
    """
    train_split_path = os.path.join(split_dir, f"{base_filename}_train_split.json")
    test_split_path = os.path.join(split_dir, f"{base_filename}_test_split.json")

    if os.path.exists(train_split_path) and os.path.exists(test_split_path):
        print(f"Found existing split files. Loading from:\n  - {train_split_path}\n  - {test_split_path}")
        with open(train_split_path, 'r') as f:
            train_indices = json.load(f)
        with open(test_split_path, 'r') as f:
            test_indices = json.load(f)
    else:
        print("No existing split found. Creating a new 50/50 split for ID data.")
        indices = id_full_df.index.astype(str).to_list()
        train_indices, test_indices = train_test_split(indices, test_size=0.5, random_state=42)

        # Special formatting for 'gta' is handled here, but the base indices are strings
        if 'gta' in base_filename:
            train_indices_to_save = [f"{int(i):05d}" for i in train_indices]
            test_indices_to_save = [f"{int(i):05d}" for i in test_indices]
        else:
            train_indices_to_save = train_indices
            test_indices_to_save = test_indices

        with open(train_split_path, 'w') as f: json.dump(train_indices_to_save, f, indent=4)
        with open(test_split_path, 'w') as f: json.dump(test_indices_to_save, f, indent=4)
        print(f"Saved new splits to:\n  - {train_split_path}\n  - {test_split_path}")

    return train_indices, test_indices

def find_best_gmm(fingerprints: pd.DataFrame, max_components: int = 11, model_type: str = ""):
    """Tests different numbers of GMM components and finds the best one using BIC."""
    data = fingerprints.to_numpy()
    bic_scores = []
    component_range = range(1, max_components + 1)
    print(f"\n--- Model Selection for {model_type.upper()} Data: Finding best GMM components ---")
    for n in component_range:
        gmm = GaussianMixture(n_components=n, random_state=42, n_init=10)
        gmm.fit(data)
        bic_scores.append(gmm.bic(data))
    best_n_components = np.argmin(bic_scores) + 1
    print(f"CONCLUSION: Optimal number of components for {model_type.upper()} is {best_n_components} (lowest BIC).")
    return best_n_components

def build_id_gmm_model(train_fingerprints: pd.DataFrame, n_components: int, model_type: str = ""):
    """Builds a GMM from the ID training fingerprints."""
    train_data = train_fingerprints.to_numpy()
    print(f"\n--- Building final GMM Model on ID-Train {model_type.upper()} data ---")
    gmm_model = GaussianMixture(n_components=n_components, random_state=42, n_init=10)
    gmm_model.fit(train_data)
    id_log_likelihoods = gmm_model.score_samples(train_data)
    max_density = np.exp(np.max(id_log_likelihoods))
    print(f"Using {n_components} component(s). Max likelihood density: {max_density:.4f}\n")
    return gmm_model, max_density

def calculate_nll_scores(fingerprints: pd.DataFrame, id_model):
    """Calculates the robust NLL-based scores."""
    gmm_model, max_density = id_model
    data = fingerprints.to_numpy()
    log_p_values = gmm_model.score_samples(data)
    ood_scores_nll = -log_p_values
    nll_min = -np.log(max_density + 1e-20)
    ood_scores_nll_zero_floored = np.maximum(0, ood_scores_nll - nll_min)
    results_df = pd.DataFrame(index=fingerprints.index)
    results_df['ood_score_nll_zero_floored'] = ood_scores_nll_zero_floored
    return results_df

def calculate_and_plot_roc(y_true, y_score, res_dir, base_filename, model_type):
    """
    Calculates AUROC, saves the ROC curve plot, and returns the AUROC score.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auroc_score = auc(fpr, tpr)
    print(f"AUROC Score ({model_type}): {auroc_score:.4f}")

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUROC = {auroc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC: {base_filename} ({model_type})'); plt.legend(loc="lower right")
    
    roc_curve_dir = os.path.join(res_dir, 'roc_curves')
    os.makedirs(roc_curve_dir, exist_ok=True)
    roc_plot_filename = os.path.join(roc_curve_dir, f"{base_filename}_{model_type}_roc_curve.png")
    plt.savefig(roc_plot_filename)
    plt.close()
    print(f"Saved ROC curve plot to: {roc_plot_filename}")
    
    return auroc_score

def run_analysis_pipeline(paths: dict, base_filename: str):
    """Loads data, handles splits, and runs the full GMM OoD analysis."""
    print("--- Loading All Datasets ---")
    id_spatial = load_spatial_fingerprints(paths['id_spatial'], base_filename)
    ood_spatial = load_spatial_fingerprints(paths['ood_spatial'], base_filename)
    id_magnitude = load_magnitude_fingerprints(paths['id_magnitude'], base_filename)
    ood_magnitude = load_magnitude_fingerprints(paths['ood_magnitude'], base_filename)

    if id_spatial is None and id_magnitude is None:
        print("Skipping analysis: No In-Distribution data found.")
        return

    if 'gta' in base_filename and id_spatial is not None:
        if pd.api.types.is_numeric_dtype(id_spatial.index):
            print("GTA dataset detected: Re-formatting spatial index to match splits (zero-padded string).")
            id_spatial.index = id_spatial.index.astype(str).str.zfill(5)

    split_basis_df = id_spatial if id_spatial is not None else id_magnitude
    save_dir = os.path.join(os.getcwd(), 'spatial', 'splits')
    os.makedirs(save_dir, exist_ok=True)
    train_indices, test_indices = get_or_create_split(split_basis_df, save_dir, base_filename)
    
    id_train_spatial = id_spatial.loc[train_indices] if id_spatial is not None else None
    id_test_spatial = id_spatial.loc[test_indices] if id_spatial is not None else None
    id_train_magnitude_raw = id_magnitude.loc[train_indices] if id_magnitude is not None else None
    id_test_magnitude_raw = id_magnitude.loc[test_indices] if id_magnitude is not None else None

    final_results_df = pd.DataFrame()
    auroc_scores_collection = {}

    # --- 1. Spatial Model ---
    if id_train_spatial is not None:
        model = build_id_gmm_model(id_train_spatial, find_best_gmm(id_train_spatial, model_type="spatial"), model_type="spatial")
        id_test_scores = calculate_nll_scores(id_test_spatial, model); id_test_scores['is_ood'] = 0
        ood_scores = calculate_nll_scores(ood_spatial, model) if ood_spatial is not None else pd.DataFrame()
        if not ood_scores.empty: ood_scores['is_ood'] = 1
        
        results = pd.concat([id_test_scores, ood_scores])
        max_nll = results['ood_score_nll_zero_floored'].max()
        final_results_df[f'ood_score_normalized_spatial'] = (results['ood_score_nll_zero_floored'] / (max_nll + 1e-9)).clip(0, 1)
        final_results_df['is_ood'] = results['is_ood']

    # --- 2. Magnitude Model (with PCA) ---
    id_train_magnitude_pca, id_test_magnitude_pca, ood_magnitude_pca = None, None, None
    if id_train_magnitude_raw is not None:
        print("\n--- Applying PCA to Magnitude Features ---")
        pca = PCA(n_components=0.95, random_state=42) # Keep components that explain 95% of variance
        id_train_magnitude_pca_data = pca.fit_transform(id_train_magnitude_raw)
        print(f"PCA selected {pca.n_components_} components for Magnitude model.")
        
        pca_cols = [f'mag_pca_{i}' for i in range(pca.n_components_)]
        id_train_magnitude_pca = pd.DataFrame(id_train_magnitude_pca_data, index=id_train_magnitude_raw.index, columns=pca_cols)

        id_test_magnitude_pca_data = pca.transform(id_test_magnitude_raw)
        id_test_magnitude_pca = pd.DataFrame(id_test_magnitude_pca_data, index=id_test_magnitude_raw.index, columns=pca_cols)
        
        if ood_magnitude is not None:
            ood_magnitude_pca_data = pca.transform(ood_magnitude)
            ood_magnitude_pca = pd.DataFrame(ood_magnitude_pca_data, index=ood_magnitude.index, columns=pca_cols)

        model = build_id_gmm_model(id_train_magnitude_pca, find_best_gmm(id_train_magnitude_pca, model_type="magnitude_pca"), model_type="magnitude_pca")
        id_test_scores = calculate_nll_scores(id_test_magnitude_pca, model)
        ood_scores = calculate_nll_scores(ood_magnitude_pca, model) if ood_magnitude_pca is not None else pd.DataFrame()
        
        results = pd.concat([id_test_scores, ood_scores])
        max_nll = results['ood_score_nll_zero_floored'].max()
        final_results_df = final_results_df.merge(
            pd.DataFrame({f'ood_score_normalized_magnitude': (results['ood_score_nll_zero_floored'] / (max_nll + 1e-9)).clip(0, 1)}),
            left_index=True, right_index=True, how='left'
        )
    
    # --- 3. Fused Model (Spatial + Magnitude-PCA) ---
    if id_train_spatial is not None and id_train_magnitude_pca is not None:
        id_train_all_fused = pd.concat([id_train_spatial, id_train_magnitude_pca], axis=1)
        id_test_all_fused = pd.concat([id_test_spatial, id_test_magnitude_pca], axis=1)
        ood_all_fused = pd.concat([ood_spatial, ood_magnitude_pca], axis=1) if ood_spatial is not None and ood_magnitude_pca is not None else None
        
        model = build_id_gmm_model(id_train_all_fused, find_best_gmm(id_train_all_fused, model_type="all_fused"), model_type="all_fused")
        id_test_scores = calculate_nll_scores(id_test_all_fused, model)
        ood_scores = calculate_nll_scores(ood_all_fused, model) if ood_all_fused is not None and not ood_all_fused.empty else pd.DataFrame()
        
        results = pd.concat([id_test_scores, ood_scores])
        max_nll = results['ood_score_nll_zero_floored'].max()
        final_results_df = final_results_df.merge(
            pd.DataFrame({f'ood_score_normalized_all': (results['ood_score_nll_zero_floored'] / (max_nll + 1e-9)).clip(0, 1)}),
            left_index=True, right_index=True, how='left'
        )

    # --- Save results and generate plots ---
    if final_results_df.empty:
        print("\nAnalysis complete, but no results were generated.")
        return

    res_dir = os.path.join(os.getcwd(), 'spatial', 'results')
    os.makedirs(res_dir, exist_ok=True)
    results_filename = os.path.join(res_dir, f"{base_filename}_scores.csv")
    final_results_df.to_csv(results_filename)
    print(f"\nSaved combined scores to:\n  - {results_filename}")

    if 'is_ood' in final_results_df.columns and 1 in final_results_df['is_ood'].unique():
        print("\n--- Generating AUROC and ROC Curve Plots ---")
        y_true = final_results_df['is_ood']
        for model_type in ['spatial', 'magnitude', 'all']:
            score_col = f'ood_score_normalized_{model_type}'
            if score_col in final_results_df.columns:
                valid_indices = final_results_df[score_col].notna()
                score = calculate_and_plot_roc(
                    y_true.loc[valid_indices], final_results_df.loc[valid_indices, score_col],
                    res_dir, base_filename, model_type
                )
                auroc_scores_collection[f'auroc_{model_type}'] = score

        if auroc_scores_collection:
            auroc_df = pd.DataFrame([auroc_scores_collection])
            auroc_csv_filename = os.path.join(res_dir, f"{base_filename}_auroc_scores.csv")
            auroc_df.to_csv(auroc_csv_filename, index=False)
            print(f"\nSaved consolidated AUROC scores to:\n  - {auroc_csv_filename}")
    else:
        print("\nSkipping AUROC calculation: No Out-of-Distribution data was processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GMM OoD analysis on spatial, magnitude, and combined fingerprints.")
    parser.add_argument('--config', type=str, default='/fast/AG_Kainmueller/vguarin/aggrigator_experiments/spatial/spatial_configs/arctique.toml', help='Path to config TOML file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    dataset_name = config['dataset']['dataset_name']
    
    print(f"Loaded configuration for dataset: '{dataset_name}'")

    analysis_paths_templates = {
        'id_spatial': config['paths'].get('id_csv_path_spatial'),
        'ood_spatial': config['paths'].get('ood_csv_path_spatial'),
        'id_magnitude': config['paths'].get('id_csv_path_magnitude'),
        'ood_magnitude': config['paths'].get('ood_csv_path_magnitude')
    }
    
    main_loop_executed = False

    if dataset_name.startswith('lidc'):
        main_loop_executed = True
        task = config['dataset']['task']
        original_variation = config['dataset']['variation']
        for var in ['malignancy', 'texture']:
            print(f"\n\n{'='*25} PROCESSING VARIATION: {var.upper()} {'='*25}")
            current_paths = {k: v.replace(original_variation, var) for k, v in analysis_paths_templates.items()}
            base_filename = f"{task}_{dataset_name}_{var}_pu"
            run_analysis_pipeline(current_paths, base_filename)
            
    if dataset_name.startswith('arctique'):
        main_loop_executed = True
        original_task = config['dataset']['task']
        original_variation = config['dataset']['variation']
        for task, var in zip(['semantic', 'instance'], ['blood_cells', 'nuclei_intensity']):
            print(f"\n\n{'='*25} PROCESSING TASK: {task.upper()}; VARIATION: {var.upper()}  {'='*25}")
            current_paths = {k: v.replace(original_task, task).replace(original_variation, var) for k, v in analysis_paths_templates.items()}
            base_filename = f"{task}_{dataset_name}_{var}_pu"
            run_analysis_pipeline(current_paths, base_filename)

    if not main_loop_executed:
        print("\n\nStandard dataset detected. Running a single analysis.")
        task = config['dataset']['task']
        variation = config['dataset']['variation']
        base_filename = f"{task}_{dataset_name}_{variation}_pu" if variation else f"{task}_{dataset_name}_pu" 
        run_analysis_pipeline(analysis_paths_templates, base_filename)

    print(f"\n{'='*25} ANALYSIS COMPLETE {'='*25}")