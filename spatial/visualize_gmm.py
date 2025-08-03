import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import argparse

def plot_gmm_ellipses(gmm, data_to_plot, feature_names, ax):
    """Plots the data and the GMM confidence ellipses."""
    
    # Plot the background data points
    ax.scatter(data_to_plot.iloc[:, 0], data_to_plot.iloc[:, 1], s=5, alpha=0.4, label='Data Points')
    
    # Define colors for the components
    colors = plt.cm.viridis(np.linspace(0, 1, gmm.n_components))

    for i in range(gmm.n_components):
        mean = gmm.means_[i]
        cov = gmm.covariances_[i]
        weight = gmm.weights_[i]
        
        # Eigen decomposition to find the ellipse's orientation and size
        vals, vecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        
        # Plot 2 standard deviations (~95% confidence)
        width, height = 2 * 2 * np.sqrt(vals)
        
        # Create and add the ellipse patch
        ell = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      facecolor=colors[i], alpha=0.5,
                      edgecolor='black', lw=1.5)
        ax.add_patch(ell)

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title("GMM Components as Confidence Ellipses")
    ax.legend()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a saved GMM model with its covariance ellipses.")
    parser.add_argument('--gmm_path', type=str, required=True, help='Path to the saved GMM .joblib file.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the corresponding training data CSV file to plot in the background.')
    parser.add_argument('--features', nargs=2, default=['moran', 'entropy'], help='Two feature names to plot.')
    
    args = parser.parse_args()

    # --- Load the Model and Data ---
    print(f"Loading GMM model from: {args.gmm_path}")
    gmm_model = joblib.load(args.gmm_path)

    print(f"Loading data from: {args.data_path}")
    # This assumes the data is in one of the original formats
    try:
        data_df = pd.read_csv(args.data_path, index_col=0)
    except Exception:
        data_df = pd.read_csv(args.data_path, index_col='uq_map_name')

    # Select the features for plotting
    plot_data = data_df[args.features]
    
    # Note: This visualizes the RAW data space. For best results, you might want to
    # apply the same transformation (e.g., BetaCDF) to the data before plotting,
    # though plotting in the original space is often more intuitive.

    # --- Create the Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Check if the loaded model is a single GMM or an ensemble
    if isinstance(gmm_model, list):
        print("Ensemble model detected. Visualizing the first model in the ensemble.")
        gmm_to_plot = gmm_model[0]
    else:
        gmm_to_plot = gmm_model

    plot_gmm_ellipses(gmm_to_plot, plot_data, args.features, ax)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()