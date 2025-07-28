import pandas as pd
import numpy as np
import toml
import os
import matplotlib.pyplot as plt
import argparse
from scipy.stats import beta, norm
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import json

class BetaCDFGaussianizer:
    """
    A class to transform features to a Gaussian distribution by fitting a Beta
    distribution and using the probability integral transform.
    This version creates a single summary plot for all features.
    """
    def __init__(self, eps=1e-10):
        self.eps = eps
        self.a_ = None
        self.b_ = None

    def fit(self, X, plot_dir=None, feature_names=None):
        """
        Fits the Beta distribution to each feature in X.

        If plot_dir is provided, it generates a single figure containing a subplot
        for each feature's distribution fit.
        """
        X_np = X.to_numpy(copy=True) if isinstance(X, pd.DataFrame) else np.asarray(X)
        X_np = np.clip(X_np, self.eps, 1 - self.eps)
        X_np = (1 - 2 * self.eps) * (X_np - 0.5) + 0.5
        self.a_ = []
        self.b_ = []

        n_features = X_np.shape[1]
        fig, axes = (None, None)
        
        if plot_dir and n_features > 0:
            os.makedirs(plot_dir, exist_ok=True)
            print(f"Generating a single feature distribution plot with {n_features} subplots in: {plot_dir}")
            
            # --- DYNAMIC SUBPLOT GRID LOGIC ---
            max_cols = 5
            n_rows = int(np.ceil(n_features / max_cols))
            n_cols = min(n_features, max_cols)
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            # Flatten the axes array for easy 1D indexing, regardless of grid shape
            flat_axes = axes.flatten()

        for i in range(n_features):
            feature_data = X_np[:, i]
            # Use the actual column name if available, otherwise default to "feature_i"
            feature_name = feature_names[i] if feature_names is not None and i < len(feature_names) else f"feature_{i}"

            if np.std(feature_data) < 1e-6:
                self.a_.append(1.0)
                self.b_.append(1.0)
            else:
                try:
                    a_hat, b_hat, _, _ = beta.fit(feature_data, floc=0, fscale=1)
                    self.a_.append(a_hat)
                    self.b_.append(b_hat)

                    if fig is not None and axes is not None:
                        # Use the flattened axis for plotting
                        ax = flat_axes[i]
                        ax.hist(feature_data, bins=30, density=True, alpha=0.6, color='g', label='Empirical')
                        x_range = np.linspace(0, 1, 1000)
                        pdf_fitted = beta.pdf(x_range, a_hat, b_hat)
                        ax.plot(x_range, pdf_fitted, 'r-', lw=2, label=f'Beta (a={a_hat:.2f}, b={b_hat:.2f})')
                        ax.set_title(f'Feature: {feature_name}')
                        ax.set_xlabel('Value')
                        ax.legend()
                except Exception as e:
                    print(f"Warning: Beta fit failed on feature '{feature_name}' with error {e}. Using defaults.")
                    self.a_.append(1.0)
                    self.b_.append(1.0)

        if fig is not None and axes is not None:
            # Add a y-label to the first plot of each row
            for r in range(n_rows):
                if r * n_cols < len(flat_axes):
                    flat_axes[r * n_cols].set_ylabel('Density')
            
            # Hide any unused subplots
            for j in range(n_features, len(flat_axes)):
                flat_axes[j].set_visible(False)

            fig.suptitle('Beta Distribution Fits for Input Features', fontsize=16)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            save_path = os.path.join(plot_dir, 'features_distribution_fit.png')
            plt.savefig(save_path)
            plt.close(fig)
            print(f"Saved feature distribution plot to: {save_path}")

        self.a_ = np.array(self.a_)
        self.b_ = np.array(self.b_)
        return self

    def transform(self, X):
        """
        Transform X using the fitted Beta parameters. Returns a DataFrame if the input was a DataFrame.
        """
        if self.a_ is None or self.b_ is None:
            raise ValueError("Must fit before transform.")

        original_index = X.index if isinstance(X, pd.DataFrame) else None
        original_columns = X.columns if isinstance(X, pd.DataFrame) else None

        X_np = X.to_numpy(copy=True) if isinstance(X, pd.DataFrame) else np.asarray(X)
        X_np = (1 - 2 * self.eps) * (X_np - 0.5) + 0.5
        # X_np = np.clip(X_np, self.eps, 1 - self.eps)
        X_transformed = np.zeros_like(X_np)

        for i in range(X_np.shape[1]):
            cdf_vals = beta.cdf(X_np[:, i], self.a_[i], self.b_[i])
            cdf_vals = (1 - 2 * self.eps) * (cdf_vals - 0.5) + 0.5
            # cdf_vals = np.clip(cdf_vals, self.eps, 1 - self.eps)
            X_transformed[:, i] = norm.ppf(cdf_vals)

        if original_index is not None and original_columns is not None:
            return pd.DataFrame(X_transformed, index=original_index, columns=original_columns)
        return X_transformed
    
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

def get_or_create_ood_split(ood_full_df: pd.DataFrame, split_dir: str, base_filename: str):
    """
    Gets or creates a 50/50 split for OOD data to ensure a balanced evaluation set.
    """
    ood_split_path = os.path.join(split_dir, f"{base_filename}_ood_split.json")
    if os.path.exists(ood_split_path):
        print(f"Found existing OOD split file. Loading from:\n  - {ood_split_path}")
        with open(ood_split_path, 'r') as f: ood_eval_indices = json.load(f)
    else:
        print("No existing OOD split found. Creating a new 15/75 split for OOD data.")
        indices = ood_full_df.index.astype(str).to_list()
        # We only need one half of the data for evaluation
        _, ood_eval_indices = train_test_split(indices, test_size=0.1, random_state=42)
        with open(ood_split_path, 'w') as f: json.dump(ood_eval_indices, f, indent=4)
        print(f"Saved new OOD split to:\n  - {ood_split_path}")
    return ood_eval_indices

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

def run_gmm_ensemble_ood_detection(train_df: pd.DataFrame, test_df: pd.DataFrame, ood_df: pd.DataFrame, n_models: int = 20, reg_covar: float = 1e-2, max_components=11):
    """Builds a GMM ensemble and calculates a robust, comparable OOD score."""
    print(f"\n--- Building GMM Ensemble ({n_models} models) on ID-Train data ---")
    print(f"Searching for optimal components (1 to {max_components}) for each model using BIC.")
    
    gmm_ensemble = []
    train_data_np = train_df.to_numpy()
    # In run_gmm_ensemble_ood_detection:
    print("Finding a single best n_components for the whole ensemble...")
    # Find best n on the full training data
    best_n_for_ensemble = find_best_gmm(train_df, max_components=max_components)

    gmm_ensemble = []
    train_data_np = train_df.to_numpy()
    
    for i in range(n_models):
        print(f"  - Training model {i+1}/{n_models}...")
        indices = np.random.choice(len(train_data_np), size=len(train_data_np), replace=True)
        bootstrap_sample = train_data_np[indices]
              
        # Use the fixed number of components
        final_gmm = GaussianMixture(n_components=best_n_for_ensemble, random_state=i, n_init=10)
        final_gmm.fit(bootstrap_sample)
        gmm_ensemble.append(final_gmm)
    
    print("\n--- Calculating Ensemble-based OOD scores ---")
    
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
    max_score = np.max(zero_floored_scores)
    std_dev_normalized = np.sqrt(variance_scores) / (max_score + 1e-9)

    # 5. Build results DataFrame
    results_df = pd.DataFrame(index=all_eval_data.index)
    results_df['ood_score_nll_zero_floored'] = zero_floored_scores
    results_df['ood_score_variance'] = variance_scores
    results_df['ood_score_std_dev_normalized'] = std_dev_normalized
    
    print(f"Ensemble's average disagreement: {((results_df['ood_score_variance'].mean()) / (results_df['ood_score_variance'].max()) * 100):.4f}%")
    print(f"Average normalized score uncertainty (Mean StdDev): {results_df['ood_score_std_dev_normalized'].mean():.4f}")

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

def plot_transformed_distributions(transformed_df, plot_dir, feature_names=None):
    """
    Plots the distributions of features after transformation and compares them
    to a standard normal distribution.
    """
    n_features = transformed_df.shape[1]
    feature_names = transformed_df.columns if feature_names is None else feature_names
    
    max_cols = 5
    n_rows = int(np.ceil(n_features / max_cols))
    n_cols = min(n_features, max_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    flat_axes = axes.flatten()

    # Create a standard normal distribution (target) for comparison
    x_range = np.linspace(-4, 4, 1000)
    pdf_normal = norm.pdf(x_range, 0, 1)

    for i in range(n_features):
        ax = flat_axes[i]
        feature_data = transformed_df.iloc[:, i].dropna()
        feature_name = feature_names[i]

        ax.hist(feature_data, bins=30, density=True, alpha=0.6, color='b', label='Transformed Data')
        ax.plot(x_range, pdf_normal, 'r-', lw=2, label='Standard Normal PDF')
        ax.set_title(f'Feature: {feature_name}')
        ax.set_xlabel('Value (Gaussian Quantiles)')
        ax.legend()
    
    # Finalize plot
    for r in range(n_rows):
        if r * n_cols < len(flat_axes):
            flat_axes[r * n_cols].set_ylabel('Density')
    
    for j in range(n_features, len(flat_axes)):
        flat_axes[j].set_visible(False)

    fig.suptitle('Feature Distributions After Gaussianization', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(plot_dir, 'features_distribution_fit_transformed.png')
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved transformed feature distribution plot to: {save_path}")

def preprocess_features(train_df, test_df, ood_df, model_type="features", res_dir=None, base_filename=None): #n_pca_components=0.95
    """
    Applies the Beta CDF Gaussianization. Learns from train_df only.
    Returns 3 processed dataframes.
    """
    print(f"\n--- Preprocessing {model_type.upper()} (Beta CDF -> Probit CDF -> Gaussian Quantiles) ---")

    gauss = BetaCDFGaussianizer()
      
    # Determine if plots should be generated
    plot_dir = None
    if res_dir and base_filename:
        plot_dir = os.path.join(res_dir, 'features_distribution', f'{base_filename}_{model_type}')

    gauss.fit(train_df.to_numpy(), plot_dir=plot_dir, feature_names=train_df.columns) #Fit Beta distribution
    
    # Transform the data
    train_processed = pd.DataFrame(gauss.transform(train_df.to_numpy()), index=train_df.index)
    test_processed = pd.DataFrame(gauss.transform(test_df.to_numpy()) , index=test_df.index)
    ood_processed = pd.DataFrame(gauss.transform(ood_df.to_numpy()), index=ood_df.index) if ood_df is not None and not ood_df.empty else None
    
    if plot_dir and train_processed is not None:
        plot_transformed_distributions(train_processed, plot_dir=plot_dir, feature_names=train_df.columns)

    return train_processed, test_processed, ood_processed 

def run_preprocessing_comparison(feature_type, id_train_df, id_test_df, ood_df, p2_n_ratio_threshold):
    """Compares different preprocessing methods for a given feature set."""
    print(f"\n--- [Comparison] Testing on: {feature_type.upper()} FEATURES ---")
    results = {}
    methods = ['beta', 'quantile', 'standardize', 'identity']
    epsilon = 1e-10
    
    for method in methods:
        print(f"  - Method: {method.upper()}")
        
        # --- PREPROCESSING LOGIC ---
        if method == 'beta':
            transformer = BetaCDFGaussianizer()
            train_proc = transformer.fit(id_train_df).transform(id_train_df)
            test_proc = transformer.transform(id_test_df)
            ood_proc = transformer.transform(ood_df) if ood_df is not None else None
        else: # For Quantile, Standardize, and Identity, apply the squeeze first
            id_train_squeezed = (1 - 2 * epsilon) * (id_train_df - 0.5) + 0.5
            id_test_squeezed = (1 - 2 * epsilon) * (id_test_df - 0.5) + 0.5
            ood_squeezed = (1 - 2 * epsilon) * (ood_df - 0.5) + 0.5 if ood_df is not None else None

            if method == 'quantile':
                transformer = QuantileTransformer(output_distribution='normal', random_state=42)
                # FIX: Use the squeezed data
                train_proc = pd.DataFrame(transformer.fit_transform(id_train_squeezed), index=id_train_df.index, columns=id_train_df.columns)
                test_proc = pd.DataFrame(transformer.transform(id_test_squeezed), index=id_test_df.index, columns=id_test_df.columns)
                ood_proc = pd.DataFrame(transformer.transform(ood_squeezed), index=ood_df.index, columns=ood_df.columns) if ood_squeezed is not None else None        
            elif method == 'standardize':
                transformer = StandardScaler()
                # FIX: Use the squeezed data
                train_proc = pd.DataFrame(transformer.fit_transform(id_train_squeezed), index=id_train_df.index, columns=id_train_df.columns)
                test_proc = pd.DataFrame(transformer.transform(id_test_squeezed), index=id_test_df.index, columns=id_test_df.columns)
                ood_proc = pd.DataFrame(transformer.transform(ood_squeezed), index=ood_df.index, columns=ood_df.columns) if ood_squeezed is not None else None  
            elif method == 'identity':
                # NEW: Handle the identity case. The "processed" data is just the squeezed data.
                train_proc = id_train_squeezed
                test_proc = id_test_squeezed
                ood_proc = ood_squeezed
            
        p_proc, n = train_proc.shape[1], len(train_proc)
        if (p_proc**2 / n) > p2_n_ratio_threshold:
            run_results = run_gmm_ensemble_ood_detection(train_proc, test_proc, ood_proc)
        else:
            n_components = find_best_gmm(train_proc, model_type=f"comparison_{feature_type}_{method}")
            gmm = build_id_gmm_model(train_proc, n_components)
            id_test_scores = calculate_nll_scores(test_proc, gmm[0], train_proc)
            ood_scores = calculate_nll_scores(ood_proc, gmm[0], train_proc) if ood_proc is not None else pd.DataFrame()
            run_results = pd.concat([id_test_scores, ood_scores])
        if ood_df is not None:
            run_results['is_ood'] = 0
            run_results.loc[ood_proc.index, 'is_ood'] = 1

        if 'is_ood' in run_results.columns and 1 in run_results['is_ood'].unique():
            fpr, tpr, _ = roc_curve(run_results['is_ood'], run_results['ood_score_nll_zero_floored'])
            results[method] = auc(fpr, tpr)
    return results

def run_full_comparison(data, p2_n_ratio_threshold, res_dir, base_filename):
    """Drives the comparison across all feature types."""
    print("\n\n" + "="*25 + " PREPROCESSING COMPARISON " + "="*25)
    full_comparison_results = pd.DataFrame(index=['beta', 'quantile', 'standardize', 'identity'])

    for feature_type in ['spatial', 'magnitude', 'all']:
        if data[f'id_train_{feature_type}'] is not None and data[f'ood_{feature_type}'] is not None:
            auroc_scores = run_preprocessing_comparison(
                feature_type,
                data[f'id_train_{feature_type}'],
                data[f'id_test_{feature_type}'],
                data[f'ood_{feature_type}'],
                p2_n_ratio_threshold
            )
            full_comparison_results[f'auroc_{feature_type}'] = pd.Series(auroc_scores)

    print("\n" + "="*25 + " COMPARISON SUMMARY " + "="*25)
    print(full_comparison_results)
    print("="*69)
    
    comp_csv_path = os.path.join(res_dir, f"{base_filename}_preprocessing_comparison.csv")
    full_comparison_results.to_csv(comp_csv_path)
    print(f"Saved full preprocessing comparison results to: {comp_csv_path}")

def run_analysis_pipeline(paths: dict, base_filename: str):
    P2_N_RATIO_THRESHOLD = 0.5
    REG_COVAR = 1e-2 # Centralize regularization parameter
    epsilon = 1e-10
    indices_were_modified = False
    
    id_spatial_raw = load_spatial_fingerprints(paths['id_spatial'], base_filename)
    ood_spatial_raw = load_spatial_fingerprints(paths['ood_spatial'], base_filename)
    id_magnitude_raw = load_magnitude_fingerprints(paths['id_magnitude'], base_filename)
    ood_magnitude_raw = load_magnitude_fingerprints(paths['ood_magnitude'], base_filename)
        
    if id_spatial_raw.index.equals(ood_spatial_raw.index):
        print(f"Indices of '{paths['id_spatial']}' and '{paths['ood_spatial']}' are identical.")
        ood_spatial_raw.index = [int(f"{idx}1") for idx in ood_spatial_raw.index]
        indices_were_modified = True
    
    if id_magnitude_raw.index.equals(ood_magnitude_raw.index):
        print(f"Indices of '{paths['id_magnitude']}' and '{paths['ood_magnitude']}' are identical.")
        ood_magnitude_raw.index = [int(f"{idx}1") for idx in ood_magnitude_raw.index]
        indices_were_modified = True

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
    
    if 'nematodes' in base_filename or 'protists' in base_filename or 'maize' in base_filename:
        ood_split_basis_df = ood_spatial_raw if ood_spatial_raw is not None else ood_magnitude_raw
        if ood_split_basis_df is not None and not ood_split_basis_df.empty:
            ood_eval_indices = get_or_create_ood_split(ood_split_basis_df, save_dir, base_filename)
            print(f"Subsampling OOD data to {len(ood_eval_indices)} points for balanced evaluation.")
            if ood_spatial_raw is not None: ood_spatial_raw = ood_spatial_raw.loc[ood_eval_indices]
            if ood_magnitude_raw is not None: ood_magnitude_raw = ood_magnitude_raw.loc[ood_eval_indices]

    data_dict = {
        'id_train_spatial': id_spatial_raw.loc[train_indices] if id_spatial_raw is not None else None,
        'id_test_spatial': id_spatial_raw.loc[test_indices] if id_spatial_raw is not None else None,
        'ood_spatial': ood_spatial_raw,
        'id_train_magnitude': id_magnitude_raw.loc[train_indices] if id_magnitude_raw is not None else None,
        'id_test_magnitude': id_magnitude_raw.loc[test_indices] if id_magnitude_raw is not None else None,
        'ood_magnitude': ood_magnitude_raw,
    }
    data_dict['id_train_all'] = pd.concat([data_dict['id_train_spatial'], data_dict['id_train_magnitude']], axis=1) if data_dict['id_train_spatial'] is not None and data_dict['id_train_magnitude'] is not None else None
    data_dict['id_test_all'] = pd.concat([data_dict['id_test_spatial'], data_dict['id_test_magnitude']], axis=1) if data_dict['id_test_spatial'] is not None and data_dict['id_test_magnitude'] is not None else None
    data_dict['ood_all'] = pd.concat([data_dict['ood_spatial'], data_dict['ood_magnitude']], axis=1).dropna() if data_dict['ood_spatial'] is not None and data_dict['ood_magnitude'] is not None else None

    res_dir = os.path.join(os.getcwd(), 'spatial', 'results')
    os.makedirs(res_dir, exist_ok=True)
    
    all_auroc_results = pd.DataFrame(index=['beta', 'quantile', 'standardize', 'identity'])
    
    # --- CORE ANALYSIS LOOP ---
    for method in ['beta', 'quantile', 'standardize', 'identity']:
        print(f"\n\n{'='*25} PROCESSING WITH METHOD: {method.upper()} {'='*25}")
        
        all_indices = split_basis_df.loc[test_indices].index.union(ood_spatial_raw.index if ood_spatial_raw is not None else pd.Index([]))
        final_results_df = pd.DataFrame(index=all_indices)
        auroc_scores_for_method = {}

        for ftype in ['spatial', 'magnitude', 'all']:
            if any(key.endswith(ftype) for key in data_dict.keys()) and data_dict[f'id_train_{ftype}'] is not None:
                print(f"\n--- Analyzing {ftype.upper()} Features (Method: {method.upper()}) ---")
                
                # --- Preprocessing ---
                id_train_df = data_dict[f'id_train_{ftype}']
                id_test_df = data_dict[f'id_test_{ftype}']
                ood_df = data_dict[f'ood_{ftype}']

                if method == 'beta':
                    train_proc, test_proc, ood_proc = preprocess_features(id_train_df, id_test_df, ood_df, model_type=f"{ftype}_{method}", res_dir=res_dir, base_filename=base_filename)
                else: # For Quantile, Standardize, and Identity, apply the squeeze first
                    id_train_squeezed = (1 - 2 * epsilon) * (id_train_df - 0.5) + 0.5
                    id_test_squeezed = (1 - 2 * epsilon) * (id_test_df - 0.5) + 0.5
                    ood_squeezed = (1 - 2 * epsilon) * (ood_df - 0.5) + 0.5 if ood_df is not None else None

                    if method == 'quantile':
                        transformer = QuantileTransformer(output_distribution='normal', random_state=42)
                        # FIX: Use the squeezed data
                        train_proc = pd.DataFrame(transformer.fit_transform(id_train_squeezed), index=id_train_df.index, columns=id_train_df.columns)
                        test_proc = pd.DataFrame(transformer.transform(id_test_squeezed), index=id_test_df.index, columns=id_test_df.columns)
                        ood_proc = pd.DataFrame(transformer.transform(ood_squeezed), index=ood_df.index, columns=ood_df.columns) if ood_squeezed is not None else None        
                    elif method == 'standardize':
                        transformer = StandardScaler()
                        # FIX: Use the squeezed data
                        train_proc = pd.DataFrame(transformer.fit_transform(id_train_squeezed), index=id_train_df.index, columns=id_train_df.columns)
                        test_proc = pd.DataFrame(transformer.transform(id_test_squeezed), index=id_test_df.index, columns=id_test_df.columns)
                        ood_proc = pd.DataFrame(transformer.transform(ood_squeezed), index=ood_df.index, columns=ood_df.columns) if ood_squeezed is not None else None  
                    elif method == 'identity':
                        # NEW: Handle the identity case. The "processed" data is just the squeezed data.
                        train_proc = id_train_squeezed
                        test_proc = id_test_squeezed
                        ood_proc = ood_squeezed
                   
                # --- OOD Detection ---
                p, n = train_proc.shape[1], len(train_proc)
                if (p**2 / n) > P2_N_RATIO_THRESHOLD:
                    results = run_gmm_ensemble_ood_detection(train_proc, test_proc, ood_proc)
                else:
                    n_components = find_best_gmm(train_proc, model_type=ftype)
                    gmm = build_id_gmm_model(train_proc, n_components)
                    id_test_scores = calculate_nll_scores(test_proc, gmm[0], train_proc)
                    id_test_scores['is_ood'] = 0
                    ood_scores = pd.DataFrame()
                    if ood_proc is not None and not ood_proc.empty:
                        ood_scores = calculate_nll_scores(ood_proc, gmm[0], train_proc)
                        ood_scores['is_ood'] = 1
                    results = pd.concat([id_test_scores, ood_scores])
                    
                max_score = results['ood_score_nll_zero_floored'].max()
                final_results_df[f'ood_score_normalized_{ftype}'] = (results['ood_score_nll_zero_floored'] / (max_score + 1e-9)).clip(0, 1)
                if 'is_ood' not in final_results_df and 'is_ood' in results:
                    final_results_df = final_results_df.merge(results[['is_ood']], left_index=True, right_index=True, how='left')
        
        # --- Save detailed scores for this method ---
        score_filename = os.path.join(res_dir, f"{base_filename}_scores_{method}.csv")
        
        if indices_were_modified:
            print("Reverting temporary index modification for CSV export...")
            # Create a copy to modify
            df_to_save = final_results_df.copy()

            # Define a safe function to revert the index
            def revert_index(idx):
                s_idx = str(idx)
                # Revert only if it's a multi-digit string ending in '1'
                if len(s_idx) > 1 and s_idx.endswith('1'):
                    # Check if the base part is a number to be safe
                    if s_idx[:-1].isdigit():
                        return int(s_idx[:-1])
                # Otherwise, return the original index
                return idx
            
            # Apply the function and save
            df_to_save.index = df_to_save.index.map(revert_index)
            df_to_save.to_csv(score_filename)
        else:
            # If no modification happened, save directly
            final_results_df.to_csv(score_filename)
            
        # final_results_df.to_csv(score_filename)
        print(f"\nSaved combined scores for {method.upper()} method to: {score_filename}")

        # --- Calculate and store AUROC scores for this method ---
        if 'is_ood' in final_results_df.columns and final_results_df['is_ood'].sum() > 0:
            y_true = final_results_df['is_ood'].dropna()
            for ftype in ['spatial', 'magnitude', 'all']:
                score_col = f'ood_score_normalized_{ftype}'
                if score_col in final_results_df.columns:
                    valid_indices = final_results_df[score_col].notna() & y_true.index.isin(final_results_df.index)
                    if y_true[valid_indices].sum() > 0:
                        model_roc_name = f"{ftype}_{method}"
                        auroc_score = calculate_and_plot_roc(y_true[valid_indices], final_results_df.loc[valid_indices, score_col], res_dir, base_filename, model_roc_name)
                        all_auroc_results.loc[method, f'auroc_{ftype}'] = auroc_score
    
    # --- Final Summary ---
    print("\n\n" + "="*25 + " FINAL COMPARISON SUMMARY " + "="*25)
    print(all_auroc_results)
    print("="*69)
    comp_csv_path = os.path.join(res_dir, f"{base_filename}_preprocessing_comparison.csv")
    all_auroc_results.to_csv(comp_csv_path)
    print(f"Saved final preprocessing comparison results to: {comp_csv_path}")

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
            
    if dataset_name.startswith('lizard'):
        main_loop_executed = True
        original_task = config['dataset']['task']
        original_variation = config['dataset']['variation']
        for task in ['semantic', 'instance']:
            print(f"\n\n{'='*25} PROCESSING TASK: {task.upper()}")
            current_paths = {k: v.replace(original_task, task) for k, v in analysis_paths_templates.items()}
            base_filename = f"{task}_{dataset_name}_{original_variation}_pu"
            run_analysis_pipeline(current_paths, base_filename)
    
    if dataset_name.startswith('wormbodies'):
        main_loop_executed = True
        original_task = config['dataset']['task']
        original_variation = config['dataset']['variation']
        for var in ['nematodes', 'protists']:
            print(f"\n\n{'='*25} PROCESSING VARIATION: {var.upper()} {'='*25}")
            current_paths = {
                k: (
                    v.replace(original_variation, var) if not (k.startswith('id') and k.endswith('spatial')) else v
                )
                for k, v in analysis_paths_templates.items()
            }
            print(current_paths)
            base_filename = f"{original_task}_{dataset_name}_{var}_pu"
            run_analysis_pipeline(current_paths, base_filename)

    if not main_loop_executed:
        print("\n\nStandard dataset detected. Running a single analysis.")
        task = config['dataset']['task']
        variation = config['dataset']['variation']
        base_filename = f"{task}_{dataset_name}_{variation}_pu" if variation else f"{task}_{dataset_name}_pu" 
        run_analysis_pipeline(analysis_paths_templates, base_filename)

    print(f"\n{'='*25} ANALYSIS COMPLETE {'='*25}")