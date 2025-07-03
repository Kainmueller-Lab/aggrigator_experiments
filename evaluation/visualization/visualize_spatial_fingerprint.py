import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

UQ_METHODS = ['dropout_pu', 'softmax_pu']
INPUT_DIR = 'output/tables'
OUTPUT_DIR = 'output/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_spatial_fingerprints(input_dir):
    """Load CSVs into a dict: {uq_method: {dataset_name: DataFrame}}"""
    data = {uq: {} for uq in UQ_METHODS}
    for fname in os.listdir(input_dir):
        if not (fname.startswith("spatial_fingerprint_") and fname.endswith(".csv")):
            continue
        parts = fname.replace('.csv', '').split('_')
        dataset_name = '_'.join(parts[2:-2])
        uq_method = '_'.join(parts[-2:])
        if uq_method not in UQ_METHODS:
            print(f"\033[93mWARNING:\033[0m Skipping {fname} (invalid UQ method)")
            continue
        try:
            df = pd.read_csv(os.path.join(input_dir, fname), header=None)
            if df.shape[1] != 4:
                print(f"Skipping {fname}: Expected 4 columns, got {df.shape[1]}")
                continue
            df.columns = ['sample_name', 'x', 'y', 'z']
            df = df.iloc[1:].copy()
            df[['x', 'y', 'z']] = df[['x', 'y', 'z']].astype(float)
            data[uq_method][dataset_name] = df
        except Exception as e:
            print(f"Failed to load {fname}: {e}")
    return data



def plot_2d_projections(data, uq_method, color_map):
    projections = [
        ('Noisy → Clustered (Moran)', 'Flat → Edge (EDS)', [0, 2]),
        ('Noisy → Clustered (Moran)', 'Constant → Diffuse (Entropy)', [0, 1]),
        ('Constant → Diffuse (Entropy)', 'Flat → Edge (EDS)', [1, 2])
    ]
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    legend_handles = {}

    for i, (ax, (xlabel, ylabel, axes_idx)) in enumerate(zip(axes, projections)):
        for dataset, df in data[uq_method].items():
            x = df.iloc[:, axes_idx[0] + 1]
            y = df.iloc[:, axes_idx[1] + 1]
            sc = ax.scatter(x, y, label=dataset, s=10, color=color_map[dataset], alpha=0.5)
            if dataset not in legend_handles:
                legend_handles[dataset] = sc
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    sorted_items = sorted(legend_handles.items())
    fig.subplots_adjust(top=0.80)
    fig.legend(
        handles=[h for _, h in sorted_items],
        labels=[l for l, _ in sorted_items],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        ncol=min(len(sorted_items), 4),
        frameon=False
    )
    plt.tight_layout(rect=[0, 0, 1, 0.78])
    out_path = os.path.join(OUTPUT_DIR, f"spatial_fingerprint_2d_{uq_method}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"2D projection plot saved to: {out_path}")
    plt.close()


def plot_3d_interactive(data, uq_method, color_map):
    fig = go.Figure()

    for dataset, df in sorted(data[uq_method].items()):
        color = color_map[dataset]
        fig.add_trace(go.Scatter3d(
            x=df['x'], y=df['y'], z=df['z'],
            mode='markers',
            marker=dict(size=2, color=color),
            name=None,
            showlegend=False,
            legendgroup=dataset,
            text=df['sample_name'],
            hovertemplate='<b>%{text}</b><br>X=%{x:.2f}<br>Y=%{y:.2f}<br>Z=%{z:.2f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            name=dataset,
            marker=dict(size=12, color=color),
            showlegend=True,
            legendgroup=dataset,
            hoverinfo='skip'
        ))

    fig.update_layout(
        title=f"3D Spatial Fingerprints ({uq_method})",
        scene=dict(
            xaxis_title="Noisy → Clustered (Moran)",
            yaxis_title="Structured → Diffuse (Entropy)",
            zaxis_title="Surface → Edge (EDS)"
        ),
        legend_title="Dataset",
        margin=dict(l=0, r=0, b=0, t=40)
    )
    out_path = os.path.join(OUTPUT_DIR, f"spatial_fingerprint_3d_{uq_method}.html")
    fig.write_html(out_path)
    print(f"3D interactive plot saved to: {out_path}")


import colorsys

def get_dataset_root(dataset_name):
    """Extracts the dataset root by splitting on numbers or known separators."""
    if dataset_name.startswith('arctique'):
        return ('_').join(dataset_name.split('_')[:2])
    if dataset_name.startswith('lidc'):
        return ('_').join(dataset_name.split('_')[:3])
    return dataset_name.split('_')[0]

def generate_color_palette(all_datasets):
    """Assign distinct colors to dataset roots, and related shades to their variations."""
    # Step 1: group variations by root
    root_to_variants = {}
    for ds in all_datasets:
        root = get_dataset_root(ds)
        root_to_variants.setdefault(root, []).append(ds)

    # Step 2: assign base colors to each root
    roots = sorted(root_to_variants)
    n_roots = len(roots)
    base_hues = [(i / n_roots) for i in range(n_roots)]
    root_to_base_color = {
        root: colorsys.hsv_to_rgb(h, 0.65, 0.95)
        for root, h in zip(roots, base_hues)
    }

    # Step 3: assign shades per variation
    dataset_to_color = {}
    for root, variants in root_to_variants.items():
        base_h, base_s, base_v = colorsys.rgb_to_hsv(*root_to_base_color[root])
        for i, ds in enumerate(sorted(variants)):
            variation_s = max(0.4, base_s - i * 0.1)
            variation_v = min(1.0, base_v - i * 0.05)
            r, g, b = colorsys.hsv_to_rgb(base_h, variation_s, variation_v)
            hex_color = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
            dataset_to_color[ds] = hex_color

    return dataset_to_color


if __name__ == "__main__":
    data = load_spatial_fingerprints(INPUT_DIR)

    # Gather dataset names across all UQ methods
    all_datasets = sorted(set(ds for uq in UQ_METHODS for ds in data[uq].keys()))

    # Create color mapping: dataset name → visually consistent color
    dataset_to_color = generate_color_palette(all_datasets)

    # Generate plots
    for uq_method in UQ_METHODS:
        if not data[uq_method]:
            print(f"\033[93mWARNING:\033[0m No data for {uq_method}, skipping.")
            continue
        plot_2d_projections(data, uq_method, dataset_to_color)
        plot_3d_interactive(data, uq_method, dataset_to_color)