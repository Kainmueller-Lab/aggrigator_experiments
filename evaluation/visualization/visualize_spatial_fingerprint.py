import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from matplotlib.colors import to_rgb 

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


import matplotlib.lines as mlines

def plot_2d_projections(data, uq_method, color_map, dataset_name):
    projections = [
        ('Noisy → Clustered (Moran)', 'Flat → Edge (EDS)', [0, 2]),
        ('Noisy → Clustered (Moran)', 'Constant → Diffuse (Entropy)', [0, 1]),
        ('Constant → Diffuse (Entropy)', 'Flat → Edge (EDS)', [1, 2])
    ]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    legend_entries = {}

    for i, (ax, (xlabel, ylabel, axes_idx)) in enumerate(zip(axes, projections)):
        for dataset, df in data[uq_method].items():
            x = df.iloc[:, axes_idx[0] + 1]
            y = df.iloc[:, axes_idx[1] + 1]
            ax.scatter(x, y, label=dataset, s=10, color=color_map[dataset], alpha=0.5)

            # Store custom legend marker
            if dataset not in legend_entries:
                legend_entries[dataset] = mlines.Line2D(
                    [], [], marker='o', linestyle='None',
                    markersize=8,  # <-- Adjust this for legend marker size
                    markerfacecolor=color_map[dataset],
                    alpha=0.8,
                    label=dataset
                )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # Sort legend entries
    sorted_items = sorted(legend_entries.items())
    legend_handles = [h for _, h in sorted_items]
    legend_labels = [l for l, _ in sorted_items]

    fig.subplots_adjust(top=0.80)
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        ncol=min(len(legend_labels), 4),
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.78])
    out_path = os.path.join(OUTPUT_DIR, f"spatial_fingerprint_2d_{dataset_name}_{uq_method}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"2D projection plot saved to: {out_path}")
    plt.close()



def plot_3d_interactive(data, uq_method, color_map, dataset_name):
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
        title=f"3D Spatial Fingerprints: {dataset_name}({uq_method})",
        scene=dict(
            xaxis_title="Noisy → Clustered (Moran)",
            yaxis_title="Structured → Diffuse (Entropy)",
            zaxis_title="Surface → Edge (EDS)"
        ),
        legend_title="Dataset",
        margin=dict(l=0, r=0, b=0, t=40)
    )
    out_path = os.path.join(OUTPUT_DIR, f"spatial_fingerprint_3d_{dataset_name}_{uq_method}.html")
    fig.write_html(out_path)
    print(f"3D interactive plot saved to: {out_path}")


import colorsys

def get_dataset_root(dataset_name):
    """Extracts the dataset root by splitting on numbers or known separators."""
    if dataset_name.startswith('arctique'):
        return ('_').join(dataset_name.split('_')[:2])
    if dataset_name.startswith('lidc'):
        return ('_').join(dataset_name.split('_')[:3])
    if dataset_name.startswith('ade20k'):
        return ('_').join(dataset_name.split('_')[:2])
    return dataset_name.split('_')[0]

def generate_color_palette(all_datasets, id_ood):
    # NOTE: If id_ood is True, then the all id sets resp. all ood sets are of same color
    # Individual datasets cannot be distinguished but we can distinguish between ID and OOD better.
    """Assign distinct colors to dataset roots, and related shades to their variations."""
    # Step 1: group variations by root
    root_to_variants = {}
    for ds in all_datasets:
        root = get_dataset_root(ds)
        root_to_variants.setdefault(root, []).append(ds)

    # Step 2: assign base colors to each root
    valid_colors = [
        'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
        'white', 'gray', 'grey', 'orange', 'purple', 'brown', 'pink',
        'lime', 'teal', 'gold', 'navy', 'maroon', 'olive'
    ]
    
    root_to_base_color = {
        'lizard': 'olive',
        'weedsgalore': 'green',
        'arctique_instance': 'cyan',
        'arctique_semantic': 'skyblue',
        'lidc_fgbg_malignancy': 'teal',
        'lidc_fgbg_texture': 'magenta',
        'ade20k_deeplabv3': 'orange',
        'ade20k_resnest': 'yellow',
        'gta': 'red'
    }

    # Step 3: assign base color or darker shade per variant
    dataset_to_color = {}
    for root, variants in root_to_variants.items():
        # Convert named base color (e.g., "blue") to RGB
        base_rgb = to_rgb(root_to_base_color[root])  # (r, g, b) in [0, 1]
        base_h, base_s, base_v = colorsys.rgb_to_hsv(*base_rgb)

        for ds in variants:
            if id_ood:
                if "0_00" in ds: # ID
                    r, g, b = to_rgb("green")
                else: # OOD
                    r, g, b = to_rgb("red")
            else:
                if "0_00" in ds or "weedsgalore" in ds:
                    # Use the base color directly
                    r, g, b = colorsys.hsv_to_rgb(base_h, base_s, base_v)
                else:
                    # Use a slightly darker variant (reduce value)
                    variation_v = max(0.0, base_v - 0.3)  # adjust as needed
                    r, g, b = colorsys.hsv_to_rgb(base_h, base_s, variation_v)

            hex_color = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
            dataset_to_color[ds] = hex_color

    return dataset_to_color



if __name__ == "__main__":
    # 1) Make overview plot of all datasets
    data = load_spatial_fingerprints(INPUT_DIR)
    show_arctique_intermediate_noise_levels = False
    show_ade20k_deeplab_only = True

    # Gather dataset names across all UQ methods
    all_datasets = sorted(set(ds for uq in UQ_METHODS for ds in data[uq].keys()))
    if not show_arctique_intermediate_noise_levels:
        all_datasets = [ds for ds in all_datasets if not 'arctique_semantic'in ds or ('arctique_semantic' in ds and ('0_00' in ds or '0_75' in ds))]
    if show_ade20k_deeplab_only:
        all_datasets = [ds for ds in all_datasets if not 'ade20k'in ds or 'ade20k' in ds and 'deeplabv3' in ds]

    plots = {'all_datasets': all_datasets}
    
    # Create color mapping: dataset name → visually consistent color
    dataset_to_color = generate_color_palette(all_datasets, id_ood=False)

    for name, datasets in plots.items():
        # Generate plots
        for uq_method in UQ_METHODS:
            if not data[uq_method]:
                print(f"\033[93mWARNING:\033[0m No data for {uq_method}, skipping.")
                continue

            # Filter only datasets that exist for this uq_method
            subset_data = {
                dataset: data[uq_method][dataset]
                for dataset in datasets
                if dataset in data[uq_method]
            }

            if not subset_data:
                print(f"\033[93mWARNING:\033[0m No matching datasets for {uq_method} in group '{name}'. Skipping.")
                continue

            plot_2d_projections({uq_method: subset_data}, uq_method, dataset_to_color, name)
            plot_3d_interactive({uq_method: subset_data}, uq_method, dataset_to_color, name)

    # 2) Make individual plots for all ID-OOD datasets
    data = load_spatial_fingerprints(INPUT_DIR)
    show_arctique_intermediate_noise_levels = False
    show_ade20k_deeplab_only = True
    # Gather dataset names across all UQ methods
    all_datasets = sorted(set(ds for uq in UQ_METHODS for ds in data[uq].keys()))
    if not show_arctique_intermediate_noise_levels:
        all_datasets = [ds for ds in all_datasets if not 'arctique_semantic'in ds or ('arctique_semantic' in ds and ('0_00' in ds or '0_75' in ds))]
    if show_ade20k_deeplab_only:
        all_datasets = [ds for ds in all_datasets if not 'ade20k'in ds or 'ade20k' in ds and 'deeplabv3' in ds]

    artique_instance_ood = sorted(set(ds for ds in all_datasets if 'arctique_instance' in ds))
    arctique_semantic_ood = sorted(set(ds for ds in all_datasets if 'arctique_semantic' in ds))
    lidc_malignancy_ood = sorted(set(ds for ds in all_datasets if 'lidc_fgbg_malignancy' in ds))
    lidc_texture_ood = sorted(set(ds for ds in all_datasets if 'lidc_fgbg_texture' in ds))
    ade20k_ood = sorted(set(ds for ds in all_datasets if 'ade20k' in ds))
    gta_ood = sorted(set(ds for ds in all_datasets if 'gta' in ds))

    plots = {'artique_instance_ood': artique_instance_ood,
            'arctique_semantic_ood': arctique_semantic_ood,
            'lidc_malignancy_ood': lidc_malignancy_ood,
            'lidc_texture_ood': lidc_texture_ood,
            'ade20k_ood': ade20k_ood,
            'gta_ood': gta_ood}
    
    # Create color mapping: dataset name → visually consistent color
    dataset_to_color = generate_color_palette(all_datasets, id_ood=True)

    for name, datasets in plots.items():
        # Generate plots
        for uq_method in UQ_METHODS:
            if not data[uq_method]:
                print(f"\033[93mWARNING:\033[0m No data for {uq_method}, skipping.")
                continue

            # Filter only datasets that exist for this uq_method
            subset_data = {
                dataset: data[uq_method][dataset]
                for dataset in datasets
                if dataset in data[uq_method]
            }

            if not subset_data:
                print(f"\033[93mWARNING:\033[0m No matching datasets for {uq_method} in group '{name}'. Skipping.")
                continue

            plot_2d_projections({uq_method: subset_data}, uq_method, dataset_to_color, name)
            plot_3d_interactive({uq_method: subset_data}, uq_method, dataset_to_color, name)