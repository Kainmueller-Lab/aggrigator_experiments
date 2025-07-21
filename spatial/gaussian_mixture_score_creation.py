import pandas as pd
import numpy as np
import toml
import os
import matplotlib.pyplot as plt
import argparse
import anndata  
import scanpy as sc  
import umap.umap_ as umap
import shutil
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from scipy.stats import norm 
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

def calculate_nll_scores(fingerprints: pd.DataFrame, gmm_model: GaussianMixture, train_fingerprints: pd.DataFrame):
    """Calculates the robust NLL-based scores for a single GMM."""
    # nll_min = -np.log(max_density + 1e-20)
    # Re-calculate max_density based on the training data provided
    train_log_likelihoods = gmm_model.score_samples(train_fingerprints.to_numpy())
    nll_min = -np.max(train_log_likelihoods)
     # Calculate scores for the new fingerprints
    data = fingerprints.to_numpy()
    log_p_values = gmm_model.score_samples(data)
    ood_scores_nll = -log_p_values
    ood_scores_nll_zero_floored = np.maximum(0, ood_scores_nll - nll_min)
    results_df = pd.DataFrame(index=fingerprints.index)
    results_df['ood_score_nll_zero_floored'] = ood_scores_nll_zero_floored
    results_df['ood_score_variance'] = 0 # Single model has 0 variance
    return results_df

### NEW: GMM Ensemble function with consistent normalization
def run_gmm_ensemble_ood_detection(train_df: pd.DataFrame, test_df: pd.DataFrame, ood_df: pd.DataFrame, n_models: int = 20, reg_covar: float = 1e-2):
    """Builds a GMM ensemble and calculates a robust, comparable OOD score."""
    print(f"\n--- Building GMM Ensemble ({n_models} models) on ID-Train data ---")
    gmm_ensemble = []
    train_data_np = train_df.to_numpy()

    for i in range(n_models):
        indices = np.random.choice(len(train_data_np), size=len(train_data_np), replace=True)
        bootstrap_sample = train_data_np[indices]
        gmm = GaussianMixture(n_components=1, random_state=i, reg_covar=reg_covar, n_init=1)
        gmm.fit(bootstrap_sample)
        gmm_ensemble.append(gmm)
    
    print("--- Calculating Ensemble-based OOD scores ---")
    
    # 1. Find the ensemble's consensus "max_density" using the training data
    train_scores_list = [gmm.score_samples(train_data_np) for gmm in gmm_ensemble]
    mean_train_log_likelihoods = np.mean(np.vstack(train_scores_list), axis=0)
    ensemble_nll_min = -np.max(mean_train_log_likelihoods)
    print(f"Ensemble's minimum NLL (anchor for normalization): {ensemble_nll_min:.4f}")

    # 2. Evaluate test and OOD data
    all_eval_data = pd.concat([test_df, ood_df] if ood_df is not None else [test_df])
    eval_scores_list = [gmm.score_samples(all_eval_data.to_numpy()) for gmm in gmm_ensemble]
    eval_scores_array = np.vstack(eval_scores_list)

    # 3. Calculate mean score and normalize it, just like the single GMM case
    mean_log_likelihoods = np.mean(eval_scores_array, axis=0)
    mean_nll_scores = -mean_log_likelihoods
    zero_floored_scores = np.maximum(0, mean_nll_scores - ensemble_nll_min)
    
    # 4. Calculate variance for diagnostics
    variance_scores = np.var(eval_scores_array, axis=0)

    # 5. Build results DataFrame
    results_df = pd.DataFrame(index=all_eval_data.index)
    results_df['ood_score_nll_zero_floored'] = zero_floored_scores
    results_df['ood_score_variance'] = variance_scores
    print(f"Ensemble's average disagreement: {((results_df['ood_score_variance'].mean()) / (results_df['ood_score_variance'].max()) * 100):.4f}%")

    results_df['is_ood'] = 0
    if ood_df is not None:
        results_df.loc[ood_df.index, 'is_ood'] = 1
            
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

def preprocess_features(train_df, test_df, ood_df, n_pca_components=.9, model_type="features"): #n_pca_components=0.95
    """
    Applies the custom Probit + Standardization + PCA transformation.
    Learns parameters ONLY from the training set.
    """
    print(f"\n--- Preprocessing {model_type.upper()} (Probit -> Standardize) ---")

    train_clipped = np.clip(train_df.to_numpy(), a_min=1e-10, a_max=1 - 1e-10)
    train_probit = norm.ppf(train_clipped)

    mean = np.mean(train_probit, axis=0)
    std = np.std(train_probit, axis=0)

    train_standardized = (train_probit - mean) / (std + 1e-9)

    # pca = PCA(n_components=n_pca_components, random_state=42)#, svd_solver='arpack')
    # train_pca_data = pca.fit_transform(train_standardized)
    # print(f"PCA selected {pca.n_components_} components, explaining {sum(pca.explained_variance_ratio_):.2f} of variance.")
    # pca_cols = [f'pca_{i}' for i in range(pca.n_components_)]
    # train_pca_df = pd.DataFrame(train_pca_data, index=train_df.index, columns=pca_cols)
    train_pca_df = pd.DataFrame(train_standardized, index=train_df.index)

    test_clipped = np.clip(test_df.to_numpy(), a_min=1e-10, a_max=1 - 1e-10)
    test_probit = norm.ppf(test_clipped)
    test_standardized = (test_probit - mean) / (std + 1e-9)
    # test_pca_data = pca.transform(test_standardized)
    # test_pca_df = pd.DataFrame(test_pca_data, index=test_df.index, columns=pca_cols)
    test_pca_df = pd.DataFrame(test_standardized, index=test_df.index)

    ood_pca_df = None
    if ood_df is not None and not ood_df.empty:
        ood_clipped = np.clip(ood_df.to_numpy(), a_min=1e-10, a_max=1 - 1e-10)
        ood_probit = norm.ppf(ood_clipped)
        ood_standardized = (ood_probit - mean) / (std + 1e-9)
        # ood_pca_data = pca.transform(ood_standardized)
        # ood_pca_df = pd.DataFrame(ood_pca_data, index=ood_df.index, columns=pca_cols)
        ood_pca_df = pd.DataFrame(ood_standardized, index=ood_df.index)

    # n_comp = pca.n_components_
    n_comp = 3
    return train_pca_df, test_pca_df, ood_pca_df, n_comp

def plot_pca_2d_visualization(id_data, ood_data, res_dir, base_filename, model_type):
    """
    Plots the first two PCA components for visualization.
    """
    if id_data is None or id_data.shape[1] < 2:
        print(f"Skipping PCA plot for {model_type}: Not enough PCA components.")
        return

    plt.figure(figsize=(10, 8))
    if id_data is not None and not id_data.empty:
        plt.scatter(id_data.iloc[:, 0], id_data.iloc[:, 1], c='blue', label='In-Distribution (Test)', alpha=0.6)
    if ood_data is not None and not ood_data.empty:
        plt.scatter(ood_data.iloc[:, 0], ood_data.iloc[:, 1], c='red', label='Out-of-Distribution (OOD)', alpha=0.6)
    plt.title(f'PCA of {model_type} Fingerprints for {base_filename}')
    plt.xlabel('Principal Component 1'); plt.ylabel('Principal Component 2')
    plt.legend(); plt.grid(True)

    vis_dir = os.path.join(res_dir, 'pca_2d_visualization') # MODIFIED
    os.makedirs(vis_dir, exist_ok=True)
    plot_filename = os.path.join(vis_dir, f"{base_filename}_{model_type}_pca.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved PCA visualization to: {plot_filename}")

def plot_umap_2d_visualization(id_data_pca, ood_data_pca, res_dir, base_filename, model_type):
    """
    Applies UMAP to PCA-transformed data and plots a 2D visualization.
    """
    print(f"Generating UMAP visualization for {model_type}...")
    if (id_data_pca is None or id_data_pca.empty) and (ood_data_pca is None or ood_data_pca.empty):
        print(f"Skipping UMAP plot for {model_type}: No data provided.")
        return

    if id_data_pca is not None and ood_data_pca is not None:
        combined_data = pd.concat([id_data_pca, ood_data_pca])
    elif id_data_pca is not None:
        combined_data = id_data_pca
    else:
        combined_data = ood_data_pca

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(combined_data)
    embedding_df = pd.DataFrame(embedding, index=combined_data.index, columns=['UMAP1', 'UMAP2'])

    plt.figure(figsize=(10, 8))
    if id_data_pca is not None:
        id_indices = id_data_pca.index
        plt.scatter(embedding_df.loc[id_indices, 'UMAP1'], embedding_df.loc[id_indices, 'UMAP2'],
                    c='blue', label='In-Distribution (Test)', alpha=0.6, s=15)
    if ood_data_pca is not None:
        ood_indices = ood_data_pca.index
        plt.scatter(embedding_df.loc[ood_indices, 'UMAP1'], embedding_df.loc[ood_indices, 'UMAP2'],
                    c='red', label='Out-of-Distribution (OOD)', alpha=0.6, s=15)
    plt.title(f'UMAP Visualization of {model_type} Fingerprints for {base_filename}')
    plt.xlabel('UMAP Dimension 1'); plt.ylabel('UMAP Dimension 2')
    plt.legend(); plt.grid(True)

    vis_dir = os.path.join(res_dir, 'umap_2d_visualization')
    os.makedirs(vis_dir, exist_ok=True)
    plot_filename = os.path.join(vis_dir, f"{base_filename}_{model_type}_umap.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved UMAP visualization to: {plot_filename}")

def plot_scanpy_pca_visualization(id_data_raw, ood_data_raw, res_dir, base_filename, pca_components, model_type):
    """
    Generates a PCA plot using the direct scanpy pipeline to replicate notebook results.
    This is for VISUALIZATION ONLY, as it processes combined ID and OOD data.
    """
    print(f"\n--- Generating visualization for {model_type} using the direct `scanpy` method ---")

    # Ensure data exists
    if id_data_raw is None or ood_data_raw is None:
        print("Skipping scanpy plot: Missing ID or OOD data.")
        return

    # --- FIX: Ensure indices are unique before processing ---
    id_data_raw.index = id_data_raw.index.astype(str)
    ood_data_raw.index = ood_data_raw.index.astype(str)
    
    # Add a prefix to OOD indices to guarantee they don't clash with ID indices
    ood_data_raw.index = "ood_" + ood_data_raw.index
    # --- END FIX ---

    # 1. Replicate the notebook's normalization process exactly
    # Learn transformation from the full ID dataset
    id_transformed = id_data_raw.copy()
    transform_params = {}
    for col in id_transformed.columns:
        # ... (rest of the function is exactly the same as before)
        data = np.clip(id_transformed[col].to_numpy(), a_min=1e-10, a_max=1 - 1e-10)
        probit_data = norm.ppf(data)
        mean, std = probit_data.mean(), probit_data.std()
        transform_params[col] = (mean, std)
        id_transformed[col] = (probit_data - mean) / (std + 1e-9)

    # Apply the same transformation to the OOD data
    ood_transformed = ood_data_raw.copy()
    for col in ood_transformed.columns:
        if col in transform_params:
            mean, std = transform_params[col]
            data = np.clip(ood_transformed[col].to_numpy(), a_min=1e-10, a_max=1 - 1e-10)
            probit_data = norm.ppf(data)
            ood_transformed[col] = (probit_data - mean) / (std + 1e-9)

    # 2. Combine all transformed data (This will now work)
    all_fingerprints_transformed = pd.concat([id_transformed, ood_transformed])

    # 3. Create the AnnData object
    adata = anndata.AnnData(all_fingerprints_transformed)
    # The sample_type logic now becomes simpler and more robust
    adata.obs['sample_type'] = ['ood' if idx.startswith('ood_') else 'id' for idx in adata.obs_names]
    
    # Ensure float32 dtype, as is standard in scanpy
    adata.X = adata.X.astype(np.float32)

    # 4. Run scanpy's PCA
    sc.pp.pca(adata, n_comps=pca_components, svd_solver='arpack', zero_center=False)
    print("Scanpy PCA computed.")

    # 4. Define filename and paths
    scanpy_filename_suffix = f"_{base_filename}_{model_type}_scanpy_pca.png"
    source_path = os.path.join('figures', f'pca{scanpy_filename_suffix}')
    vis_dir = os.path.join(res_dir, 'pca_2d_visualization')
    os.makedirs(vis_dir, exist_ok=True)
    destination_path = os.path.join(vis_dir, f"{base_filename}_{model_type}_scanpy_pca.png")

    try:
        # 5. Plot using scanpy, saving with only the unique suffix
        sc.pl.pca(
            adata,
            color='sample_type',
            title=f'Normalized and dim-reduced features ({model_type})',
            save=scanpy_filename_suffix,
            show=False
        )
        
        # 6. Move the file from the scanpy default dir to our desired dir
        shutil.move(source_path, destination_path)
        print(f"Saved and moved scanpy PCA visualization to: {destination_path}")

    except FileNotFoundError:
        print(f"Error: Could not find the saved scanpy plot at '{source_path}'. The plot might not have been generated.")
    
    finally:
        # --- NEW CLEANUP BLOCK ---
        # 7. Clean up the temporary 'figures' directory created by scanpy
        if os.path.exists('figures'):
            shutil.rmtree('figures')
            print("Cleaned up temporary 'figures' directory.")
        # --- END CLEANUP BLOCK ---
        plt.close()

def run_analysis_pipeline(paths: dict, base_filename: str):
    P2_N_RATIO_THRESHOLD = 0.5
    REG_COVAR = 1e-2 # Centralize regularization parameter
    
    id_spatial_raw = load_spatial_fingerprints(paths['id_spatial'], base_filename)
    ood_spatial_raw = load_spatial_fingerprints(paths['ood_spatial'], base_filename)
    id_magnitude_raw = load_magnitude_fingerprints(paths['id_magnitude'], base_filename)
    ood_magnitude_raw = load_magnitude_fingerprints(paths['ood_magnitude'], base_filename)
        
    if id_spatial_raw.index.equals(ood_spatial_raw.index):
        print(f"Indices of '{paths['id_spatial']}' and '{paths['ood_spatial']}' are identical.")
        ood_spatial_raw.index = [int(f"{idx}1") for idx in ood_spatial_raw.index]
    
    if id_magnitude_raw.index.equals(ood_magnitude_raw.index):
        print(f"Indices of '{paths['id_magnitude']}' and '{paths['ood_magnitude']}' are identical.")
        ood_magnitude_raw.index = [int(f"{idx}1") for idx in ood_magnitude_raw.index]

    if id_spatial_raw is None and id_magnitude_raw is None:
        print("Skipping analysis: No In-Distribution data found.")
        return

    if 'gta' in base_filename and id_spatial_raw is not None:
        if pd.api.types.is_numeric_dtype(id_spatial_raw.index):
            print("GTA dataset detected: Re-formatting spatial index to match splits (zero-padded string).")
            id_spatial_raw.index = id_spatial_raw.index.astype(str).str.zfill(5)
    
    split_basis_df = id_spatial_raw if id_spatial_raw is not None else id_magnitude_raw
    save_dir = os.path.join(os.getcwd(), 'spatial', 'splits')
    os.makedirs(save_dir, exist_ok=True)
    train_indices, test_indices = get_or_create_split(split_basis_df, save_dir, base_filename)
    
    id_train_spatial = id_spatial_raw.loc[train_indices] if id_spatial_raw is not None else None
    id_test_spatial = id_spatial_raw.loc[test_indices] if id_spatial_raw is not None else None
    id_train_magnitude = id_magnitude_raw.loc[train_indices] if id_magnitude_raw is not None else None
    id_test_magnitude = id_magnitude_raw.loc[test_indices] if id_magnitude_raw is not None else None

    final_results_df = pd.DataFrame()
    auroc_scores_collection = {}
    res_dir = os.path.join(os.getcwd(), 'spatial', 'results')

    # --- 1. Spatial Model ---
    if id_train_spatial is not None:
        id_train_spat_pca, id_test_spat_pca, ood_spat_pca = id_train_spatial, id_test_spatial, ood_spatial_raw
        # id_train_spat_pca, id_test_spat_pca, ood_spat_pca, _ = preprocess_features(
        #     id_train_spatial, id_test_spatial, ood_spatial_raw, model_type="spatial"
        # )
        model = build_id_gmm_model(id_train_spat_pca, find_best_gmm(id_train_spat_pca, model_type="spatial"), model_type="spatial")
        id_test_scores = calculate_nll_scores(id_test_spat_pca, model[0], id_train_spat_pca); id_test_scores['is_ood'] = 0
        ood_scores = calculate_nll_scores(ood_spat_pca, model[0], id_train_spat_pca) if ood_spat_pca is not None else pd.DataFrame()
        if not ood_scores.empty: ood_scores['is_ood'] = 1
        
        results = pd.concat([id_test_scores, ood_scores])
        max_nll = results['ood_score_nll_zero_floored'].max()
        final_results_df['ood_score_normalized_spatial'] = (results['ood_score_nll_zero_floored'] / (max_nll + 1e-9)).clip(0, 1)
        final_results_df['is_ood'] = results['is_ood']

    # --- 2. Magnitude Model (with custom preprocessing and adaptive GMM / SVM) ---
    id_train_magnitude_pca, id_test_magnitude_pca, ood_magnitude_pca = None, None, None
    if id_train_magnitude is not None:
        p = id_train_magnitude.shape[1]
        n = id_train_magnitude.shape[0]
        p2_n_ratio = (p**2) / n
        print(f"\n--- Magnitude Model Analysis ---")
        print(f"Features (p): {p}, Train Samples (n): {n}, p²/n Ratio: {p2_n_ratio:.4f}")

        id_train_magnitude_pca, id_test_magnitude_pca, ood_magnitude_pca, _ = preprocess_features(
            id_train_magnitude, id_test_magnitude, ood_magnitude_raw, model_type="magnitude"
        )
        
        # Re-evaluate p based on processed data
        p_proc = id_train_magnitude_pca.shape[1]
        p2_n_ratio_proc = (p_proc**2) / n
        
        if p2_n_ratio_proc > P2_N_RATIO_THRESHOLD:
            print(f"WARNING: p²/n ratio ({p2_n_ratio:.4f}) is high. Switching to GMM Ensemble.")
            results = run_gmm_ensemble_ood_detection(id_train_magnitude_pca, id_test_magnitude_pca, ood_magnitude_pca)
        else:
            print(f"p²/n ratio ({p2_n_ratio_proc:.4f}) is acceptable. Using GMM.")
            model = build_id_gmm_model(id_train_magnitude_pca, find_best_gmm(id_train_magnitude_pca, model_type="magnitude_pca"), model_type="magnitude_pca")
            id_test_scores = calculate_nll_scores(id_test_magnitude_pca, model[0], id_train_magnitude_pca)
            ood_scores = calculate_nll_scores(ood_magnitude_pca, model[0], id_train_magnitude_pca) if ood_magnitude_pca is not None else pd.DataFrame()
            if not ood_scores.empty: ood_scores['is_ood'] = 1
            results = pd.concat([id_test_scores, ood_scores])
        
            # --- PLOT PCA & UMAP VISUALIZATIONS ---
            # plot_pca_2d_visualization(id_test_magnitude_pca, ood_magnitude_pca, res_dir, base_filename, model_type="magnitude")
            # plot_umap_2d_visualization(id_test_magnitude_pca, ood_magnitude_pca, res_dir, base_filename, model_type="magnitude")
            # --- END PLOTS ---
        
        max_score = results['ood_score_nll_zero_floored'].max()
        final_results_df = final_results_df.merge(
            pd.DataFrame({'ood_score_normalized_magnitude': (results['ood_score_nll_zero_floored'] / (max_score + 1e-9)).clip(0, 1)}),
            left_index=True, right_index=True, how='left'
        )
    
    # --- 3. Fused Model (Spatial + Magnitude -> Preprocess -> GMM) ---
    if id_train_spatial is not None and id_train_magnitude is not None:
        # Concatenate RAW features first
        id_train_all_raw = pd.concat([id_train_spatial, id_train_magnitude], axis=1)
        id_test_all_raw = pd.concat([id_test_spatial, id_test_magnitude], axis=1)

        ood_all_raw = None
        if ood_spatial_raw is not None and ood_magnitude_raw is not None:
            common_ood_indices = ood_spatial_raw.index.intersection(ood_magnitude_raw.index)
            if not common_ood_indices.empty:
                ood_all_raw = pd.concat([ood_spatial_raw.loc[common_ood_indices], ood_magnitude_raw.loc[common_ood_indices]], axis=1)
                
        p = id_train_all_raw.shape[1]
        n = id_train_all_raw.shape[0]
        p2_n_ratio = (p**2) / n
        print(f"\n--- Fused Model Analysis ---")
        print(f"Features (p): {p}, Train Samples (n): {n}, p²/n Ratio: {p2_n_ratio:.4f}")

        # A bis) Apply the rigorous pipeline for the real model and its visualization
        id_train_all_pca, id_test_all_pca, ood_all_pca, pca_n_components = preprocess_features(
            id_train_all_raw, id_test_all_raw, ood_all_raw, model_type="all_fused"
        )

        # Add visualizations for the fused data
        # plot_pca_2d_visualization(id_test_all_pca, ood_all_pca, res_dir, base_filename, model_type="all_fused")
        # plot_umap_2d_visualization(id_test_all_pca, ood_all_pca, res_dir, base_filename, model_type="all_fused")
        
        # --- VISUALIZATION BLOCK ---
        # A) Use direct scanpy method for the explanatory plot
        # We use the full ID and OOD sets here to perfectly replicate the notebook
        full_id_all_raw = pd.concat([id_spatial_raw, id_magnitude_raw], axis=1).dropna()
        full_ood_fused = None
        if ood_spatial_raw is not None and ood_magnitude_raw is not None:
            full_ood_fused = pd.concat([ood_spatial_raw, ood_magnitude_raw], axis=1).dropna()
        plot_scanpy_pca_visualization(full_id_all_raw, full_ood_fused, res_dir, base_filename, pca_n_components, model_type="all_fused")
        
        # Your original, correct plots
        # plot_pca_2d_visualization(id_test_all_pca, ood_all_pca, res_dir, base_filename, model_type="all_fused_rigorous")
        # plot_umap_2d_visualization(id_test_all_pca, ood_all_pca, res_dir, base_filename, model_type="all_fused_rigorous")
        # --- END VISUALIZATION BLOCK ---
        
        p_proc = id_train_all_pca.shape[1]
        p2_n_ratio_proc = (p_proc**2) / n

        if p2_n_ratio_proc > P2_N_RATIO_THRESHOLD:
            print(f"WARNING: p²/n ratio ({p2_n_ratio_proc:.4f}) is high. Switching to GMM Ensemble.")
            results = run_gmm_ensemble_ood_detection(id_train_all_pca, id_test_all_pca, ood_all_pca)
        else:
            print(f"p²/n ratio ({p2_n_ratio_proc:.4f}) is acceptable. Using GMM.")
            # Build GMM on the processed fused data
            model = build_id_gmm_model(id_train_all_pca, find_best_gmm(id_train_all_pca, model_type="all_fused_pca"), model_type="all_fused_pca")
            id_test_scores = calculate_nll_scores(id_test_all_pca, model[0], id_train_all_pca)
            ood_scores = calculate_nll_scores(ood_all_pca, model[0], id_train_all_pca) if ood_all_pca is not None and not ood_all_pca.empty else pd.DataFrame()
            if not ood_scores.empty: ood_scores['is_ood'] = 1
            results = pd.concat([id_test_scores, ood_scores])
            
        max_score = results['ood_score_nll_zero_floored'].max()
        final_results_df = final_results_df.merge(
            pd.DataFrame({'ood_score_normalized_all': (results['ood_score_nll_zero_floored'] / (max_score + 1e-9)).clip(0, 1)}),
            left_index=True, right_index=True, how='left'
        )
    
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
        original_noise = config['dataset']['noise_level']
        for task, var, ns in zip(['semantic', 'instance'], ['blood_cells', 'nuclei_intensity'], ['0_75', '0_50']):
            print(f"\n\n{'='*25} PROCESSING TASK: {task.upper()}; VARIATION: {var.upper()}  {'='*25}")
            current_paths = {
                k: (
                    v.replace(original_task, task)
                    .replace(original_variation, var)
                    .replace(original_noise, ns) if k.startswith('ood')
                    else v.replace(original_task, task).replace(original_variation, var)
                )
                for k, v in analysis_paths_templates.items()
            }
            base_filename = f"{task}_{dataset_name}_{var}_pu"
            run_analysis_pipeline(current_paths, base_filename)

    if not main_loop_executed:
        print("\n\nStandard dataset detected. Running a single analysis.")
        task = config['dataset']['task']
        variation = config['dataset']['variation']
        base_filename = f"{task}_{dataset_name}_{variation}_pu" if variation else f"{task}_{dataset_name}_pu" 
        run_analysis_pipeline(analysis_paths_templates, base_filename)

    print(f"\n{'='*25} ANALYSIS COMPLETE {'='*25}")