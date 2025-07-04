import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import colormaps

INPUT_DIR = 'output/tables'
OUTPUT_DIR = 'output/figures'
UQ_METHODS = ['dropout_pu']
AGG_METHOD = 'patch_aggregation_20' 

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_fingerprints(input_dir):
    """Load spatial_fingerprint_*.csv files."""
    data = {uq: {} for uq in UQ_METHODS}
    for fname in os.listdir(input_dir):
        if not fname.startswith("spatial_fingerprint_") or not fname.endswith(".csv"):
            continue
        parts = fname.replace('.csv', '').split('_')
        dataset = '_'.join(parts[2:-2])
        uq = '_'.join(parts[-2:])
        if uq not in UQ_METHODS:
            continue
        try:
            df = pd.read_csv(os.path.join(input_dir, fname), header=None)
            if df.shape[1] != 4:
                continue
            df.columns = ['sample_name', 'x', 'y', 'z']
            df = df.iloc[1:].copy()
            df[['x', 'y', 'z']] = df[['x', 'y', 'z']].astype(float)
            data[uq][dataset] = df
        except Exception as e:
            print(f"Failed to load {fname}: {e}")
    return data

def load_aggregation_values(dataset, uq_method, method_name):
    fname = f"aggregation_value_summary_{dataset}_{uq_method}.csv"
    fpath = os.path.join(INPUT_DIR, fname)
    if not os.path.exists(fpath):
        print(f"\033[93mWARNING:\033[0m Missing summary file: {fname}")
        return None
    try:
        df = pd.read_csv(fpath)
        values = df[method_name].values
        if len(values) > 0 and not np.issubdtype(values.dtype, np.number):
            values = values[1:]  # skip header row if misread
        return values.astype(float)
    except Exception as e:
        print(f"\033[91mERROR:\033[0m Reading {fname}: {e}")
        return None

def plot_3d_by_value(data, uq_method, method_name, dataset):
    fig = go.Figure()

    for dataset, coords in data[uq_method].items():
        values = load_aggregation_values(dataset, uq_method, method_name)
        if values is None or len(values) != len(coords):
            print(f"\033[93mWARNING:\033[0m Value mismatch for {dataset}, skipping.")
            continue

        fig.add_trace(go.Scatter3d(
            x=coords['x'], y=coords['y'], z=coords['z'],
            mode='markers',
            marker=dict(
                size=2,
                color=values,
                colorscale='Reds',
                cmin=min(values),
                cmax=max(values),
                colorbar=dict(title=method_name)
            ),
            text=coords['sample_name'],
            name=dataset,
            hovertemplate='<b>%{text}</b><br>X=%{x:.2f}<br>Y=%{y:.2f}<br>Z=%{z:.2f}<br>' +
                          f'{method_name}=%{{marker.color:.2f}}<extra></extra>'
        ))

    fig.update_layout(
        title=f"3D Fingerprint Colored by {method_name} ({uq_method})",
        scene=dict(
            xaxis_title="Moran",
            yaxis_title="Entropy",
            zaxis_title="EDS"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    outpath = os.path.join(OUTPUT_DIR, f"spatial_fingerprint_3d_{dataset}_{uq_method}_{method_name}.html")
    fig.write_html(outpath)
    print(f"3D plot saved to: {outpath}")


def plot_2d_by_value(data, uq_method, method_name, dataset):
    projections = [
        ('Moran', 'EDS', [0, 2]),
        ('Moran', 'Entropy', [0, 1]),
        ('Entropy', 'EDS', [1, 2])
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=False)
    cmap = colormaps['Reds']

    all_values = []

    for dataset, coords in data[uq_method].items():
        values = load_aggregation_values(dataset, uq_method, method_name)
        if values is None or len(values) != len(coords):
            continue
        all_values.extend(values)

    norm = Normalize(vmin=min(all_values), vmax=max(all_values))

    for ax, (xlabel, ylabel, (i, j)) in zip(axes, projections):
        for dataset, coords in data[uq_method].items():
            values = load_aggregation_values(dataset, uq_method, method_name)
            if values is None or len(values) != len(coords):
                continue
            x = coords.iloc[:, i+1]
            y = coords.iloc[:, j+1]
            ax.scatter(x, y, c=values, cmap=cmap, norm=norm, s=10, alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # Create horizontal colorbar above the subplots
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # Adjust space to make room for colorbar
    fig.subplots_adjust(top=0.85, bottom=0.1)
    cbar_ax = fig.add_axes([0.25, 0.88, 0.5, 0.03])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(method_name)

    fig.suptitle(f"2D Spatial fingerprint of {dataset} colored by {method_name} ({uq_method})", fontsize=14)
    outpath = os.path.join(OUTPUT_DIR, f"spatial_fingerprint_2d_by_aggregation_{dataset}_{uq_method}_{method_name}.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"2D plot saved to: {outpath}")
    plt.close()



import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize aggregation summary projections")
    parser.add_argument("--aggregation", type=str, help="Name of aggregation strategy.")
    parser.add_argument("--dataset", type=str, help="name of id/ood dataset root. Options: ade20k, arctique_semantic, arctique_instance, lidc_malignancy, lidc_texture, gta")
    args = parser.parse_args()

    AGG_METHOD = args.aggregation
    dataset = args.dataset

    data = load_fingerprints(INPUT_DIR)

    # Define groupings
    datasets_to_plot = {
        #'all_datasets': [ds for uq in UQ_METHODS for ds in data[uq]], # NOTE: Uncomment this if you want to plot all datasets
        'arctique_semantic': [ds for uq in UQ_METHODS for ds in data[uq] if 'arctique_semantic' in ds],
        'arctique_instance': [ds for uq in UQ_METHODS for ds in data[uq] if 'arctique_instance' in ds],
        'lidc_malignancy': [ds for uq in UQ_METHODS for ds in data[uq] if 'lidc_fgbg_malignancy' in ds],
        'lidc_texture': [ds for uq in UQ_METHODS for ds in data[uq] if 'lidc_fgbg_texture' in ds],
        'ade20k': [ds for uq in UQ_METHODS for ds in data[uq] if 'ade20k' in ds],
        'gta': [ds for uq in UQ_METHODS for ds in data[uq] if 'gta' in ds]
    }

    if dataset not in datasets_to_plot:
        print(f"\033[91mERROR:\033[0m Unknown dataset group: {dataset}")
        print(f"Available groups: {', '.join(datasets_to_plot.keys())}")
        exit(1)

    group_datasets = datasets_to_plot[dataset]

    for uq_method in UQ_METHODS:
        subset = {
            uq_method: {
                ds: data[uq_method][ds]
                for ds in group_datasets
                if ds in data[uq_method]
            }
        }

        if not subset[uq_method]:
            print(f"\033[93mWARNING:\033[0m No data for group {dataset}, method {uq_method}")
            continue

        plot_2d_by_value(subset, uq_method, AGG_METHOD, dataset)
        # NOTE: Uncomment this if 3d plot is necessary.
        # plot_3d_by_value(subset, uq_method, AGG_METHOD, dataset)

