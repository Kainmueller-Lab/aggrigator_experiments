
import shutil
import anndata
import scanpy as sc  
import umap.umap_ as umap
from scipy.stats import norm 
from sklearn.decomposition import PCA

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

    vis_dir = os.path.join(res_dir, 'pca_2d_visualization') 
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

    # --- Ensure indices are unique before processing ---
    id_data_raw.index = id_data_raw.index.astype(str)
    ood_data_raw.index = ood_data_raw.index.astype(str)
    
    # Add a prefix to OOD indices to guarantee they don't clash with ID indices
    ood_data_raw.index = "ood_" + ood_data_raw.index

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
        # 7. Clean up the temporary 'figures' directory created by scanpy
        if os.path.exists('figures'):
            shutil.rmtree('figures')
            print("Cleaned up temporary 'figures' directory.")
        plt.close()
