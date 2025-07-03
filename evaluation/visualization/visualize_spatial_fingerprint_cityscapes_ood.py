import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

INPUT_DIR = 'output/tables'
OUTPUT_DIR = 'output/figures'
UQ_METHODS = ['dropout_pu']

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_cityscapes_variants(input_dir):
    """Load CSVs from all cityscapes* datasets only"""
    data = {uq: {} for uq in UQ_METHODS}
    for fname in os.listdir(input_dir):
        if not (fname.startswith("spatial_fingerprint_") and fname.endswith(".csv")):
            continue
        parts = fname.replace('.csv', '').split('_')
        dataset_name = '_'.join(parts[2:-2])
        dataset_root = parts[2]
        uq_method = '_'.join(parts[-2:])
        if not any(w in dataset_name for w in ["cityscapes", "gta", "ade20k"]):
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

def assign_distinct_colors(datasets):
    """Assign a distinct color to each dataset name."""
    palette = px.colors.qualitative.Alphabet  # 26+ distinguishable colors
    return {
        ds: palette[i % len(palette)]
        for i, ds in enumerate(sorted(datasets))
    }

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
    out_path = os.path.join(OUTPUT_DIR, f"spatial_fingerprint_2d_cityscapes_texture_ood_{uq_method}.png")
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
        title=f"3D Spatial Fingerprints (cityscapes, {uq_method})",
        scene=dict(
            xaxis_title="Noisy → Clustered (Moran)",
            yaxis_title="Structured → Diffuse (Entropy)",
            zaxis_title="Surface → Edge (EDS)"
        ),
        legend_title="Dataset",
        margin=dict(l=0, r=0, b=0, t=40)
    )
    out_path = os.path.join(OUTPUT_DIR, f"spatial_fingerprint_3d_cityscapes_texture_ood_{uq_method}.html")
    fig.write_html(out_path)
    print(f"3D interactive plot saved to: {out_path}")


if __name__ == "__main__":
    data = load_cityscapes_variants(INPUT_DIR)
    all_datasets = sorted(set(ds for uq in UQ_METHODS for ds in data[uq].keys()))
    if not all_datasets:
        print("No cityscapes datasets found.")
    else:
        color_map = assign_distinct_colors(all_datasets)
        for uq in UQ_METHODS:
            if not data[uq]:
                print(f"\033[93mWARNING:\033[0m No data for {uq}, skipping.")
                continue
            plot_2d_projections(data, uq, color_map)
            plot_3d_interactive(data, uq, color_map)
