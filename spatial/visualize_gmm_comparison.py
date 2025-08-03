import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import joblib
import argparse

def plot_gmm_ellipses(ax, gmm_model, id_data, ood_data, features, title):
    """
    Plots the GMM confidence ellipses and the data points on a given axis.
    This function intelligently slices the high-dimensional GMM parameters
    to visualize them in 2D.
    """
    
    # --- 1. Find the indices of the features we want to plot ---
    # This is crucial for slicing the 21-dimensional model parameters
    try:
        feature_indices = [list(id_data.columns).index(f) for f in features]
    except ValueError as e:
        print(f"Error: Feature '{e.args[0].split(' ')[0]}' not found in the data columns.")
        return

    # --- 2. Plot the background data points ---
    ax.scatter(
        id_data[features[0]], id_data[features[1]],
        s=10, alpha=0.5, c='royalblue', label='In-Distribution (ID)'
    )
    ax.scatter(
        ood_data[features[0]], ood_data[features[1]],
        s=15, alpha=0.8, c='red', marker='x', label='Out-of-Distribution (OOD)'
    )

    # --- 3. Plot the GMM Ellipses ---
    for i in range(gmm_model.n_components):
        # Extract the 2D mean and 2x2 covariance matrix for the selected features
        mean_2d = gmm_model.means_[i, feature_indices]
        cov_full = gmm_model.covariances_[i]
        # This powerful numpy indexing creates the 2x2 covariance matrix
        cov_2d = cov_full[np.ix_(feature_indices, feature_indices)]

        # Eigen decomposition to find the ellipse's orientation and size
        vals, vecs = np.linalg.eigh(cov_2d)
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        
        # Plot 2 standard deviations (~95% confidence)
        width, height = 2 * 2 * np.sqrt(vals)
        
        ell = Ellipse(
            xy=mean_2d, width=width, height=height, angle=angle,
            facecolor='gray', alpha=0.35, edgecolor='black', lw=1.5,
            zorder=0 # Send ellipse to the back
        )
        ax.add_patch(ell)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    # Using 'equal' aspect ratio is VITAL to correctly see the shape of the ellipses
    ax.set_aspect('equal', 'box')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a side-by-side comparison plot of two GMMs to visualize their support regions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Arguments for the LIZARD model ---
    parser.add_argument('--lizard-gmm', default='/fast/AG_Kainmueller/vguarin/aggrigator_experiments/spatial/results/gmm_models/semantic_lizard_glas_set_pu_spatial_standardize_model.joblib', help='Path to the saved LIZARD GMM .joblib file')
    parser.add_argument('--lizard-id-data', default='/fast/AG_Kainmueller/vguarin/aggrigator_experiments/output/tables/spatial_fingerprint_lizard_semantic_glas_set_0_00_dropout_pu.csv', help='Path to the LIZARD in-distribution data CSV')
    parser.add_argument('--lizard-ood-data',  default='/fast/AG_Kainmueller/vguarin/aggrigator_experiments/output/tables/spatial_fingerprint_lizard_semantic_glas_set_1_00_dropout_pu.csv',  help='Path to the LIZARD out-of-distribution data CSV')
    
    # --- Arguments for the GTA model ---
    parser.add_argument('--gta-gmm', default='/fast/AG_Kainmueller/vguarin/aggrigator_experiments/spatial/results/gmm_models/semantic_gta_cityscapes_pu_standardize_spatial_model.joblib', help='Path to the saved GTA GMM .joblib file')
    parser.add_argument('--gta-id-data', default='/fast/AG_Kainmueller/vguarin/aggrigator_experiments/output/tables/spatial_fingerprint_gta_0_00_dropout_pu.csv', help='Path to the GTA in-distribution data CSV')
    parser.add_argument('--gta-ood-data', default='/fast/AG_Kainmueller/vguarin/aggrigator_experiments/output/tables/spatial_fingerprint_gta_1_00_dropout_pu.csv', help='Path to the GTA out-of-distribution data CSV')
    
    # --- General plotting arguments ---
    parser.add_argument('--features', nargs=2, default=['eds', 'moran'], help='Two feature names to plot')
    parser.add_argument('--output-path', default='/fast/AG_Kainmueller/vguarin/aggrigator_experiments/spatial/results/ellipsoids/gmm_spat_liz_vs_gta_comparison_plot.png', help='Path to save the final plot')
    
    args = parser.parse_args()

    # --- Load all data and models ---
    print("Loading data and models...")
    lizard_gmm = joblib.load(args.lizard_gmm)
    gta_gmm = joblib.load(args.gta_gmm)
    
    lizard_id_df = pd.read_csv(args.lizard_id_data, index_col=0)
    lizard_ood_df = pd.read_csv(args.lizard_ood_data, index_col=0)
    gta_id_df = pd.read_csv(args.gta_id_data, index_col=0)
    gta_ood_df = pd.read_csv(args.gta_ood_data, index_col=0)
    
    # Create the figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle(f'GMM Support Region Comparison: {args.features[0]} vs. {args.features[1]}', fontsize=18)

    # --- Plot 1: LIZARD ---
    print("Plotting LIZARD GMM...")
    plot_gmm_ellipses(
        axes[0], lizard_gmm, lizard_id_df, lizard_ood_df, args.features,
        'LIZARD: Shrunk Volume (Poor OOD Detection)'
    )

    # --- Plot 2: GTA ---
    print("Plotting GTA GMM...")
    plot_gmm_ellipses(
        axes[1], gta_gmm, gta_id_df, gta_ood_df, args.features,
        'GTA: Wide Volume (Good OOD Detection)'
    )

    # Final adjustments and saving
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.output_path, dpi=150)
    print(f"\nPlot saved successfully to: {args.output_path}")
    plt.show()