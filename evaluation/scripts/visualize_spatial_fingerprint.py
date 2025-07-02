import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import re
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px



def plot_2d_projections(directory):
    pattern = re.compile(r"spatial_fingerprint_(.+?)\.csv")
    datasets = {}

    for file in os.listdir(directory):
        match = pattern.match(file)
        if match:
            identifier = match.group(1)
            filepath = os.path.join(directory, file)
            try:
                df = pd.read_csv(filepath, header=None)
                if df.shape[1] != 4:
                    print(f"Skipping {file}: Expected 4 columns, got {df.shape[1]}")
                    continue
                # Drop filename column, keep only x, y, z
                coords = df.iloc[:, 1:4]
                # Drop first row (header)
                coords = coords.iloc[1:]
                datasets[identifier] = coords.astype(float)
            except Exception as e:
                print(f"Failed to load {file}: {e}")

    # Plot 2D projections
    projections = [
        ('Noisy --> Clustered Uncertainty (Moran)', 'Flat --> Edge Uncertainty (EDS)', [0, 2]),
        ('Noisy --> Clustered Uncertainty (Moran)', 'Constant --> Diffuse Uncertainty (Entropy)', [0, 1]),
        ('Constant --> Diffuse Uncertainty (Entropy)', 'Flat --> Edge Uncertainty (EDS)', [1, 2])
    ]

    # Create figure and 3 horizontal subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    # Collect legend handles only once
    legend_handles = {}

    for i, (ax, (xlabel, ylabel, axes_idx)) in enumerate(zip(axes, projections)):
        for identifier, coords in datasets.items():
            x = coords.iloc[:, axes_idx[0]]
            y = coords.iloc[:, axes_idx[1]]
            sc = ax.scatter(x, y, label=identifier, s=10)
            if identifier not in legend_handles:
                legend_handles[identifier] = sc
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # Sort legend entries alphabetically by label
    sorted_items = sorted(legend_handles.items(), key=lambda item: item[0])
    sorted_labels = [label for label, _ in sorted_items]
    sorted_handles = [handle for _, handle in sorted_items]

    # Format legend
    ncol = min(len(sorted_labels), 4)
    fig.subplots_adjust(top=0.80)
    fig.legend(
        handles=sorted_handles,
        labels=sorted_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        ncol=ncol,
        frameon=False
    )

    # Layout adjustment
    plt.tight_layout(rect=[0, 0, 1, 0.78])

    # Save figure
    output_path = os.path.join("output", "figures", "spatial_fingerprint_pointcloud.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined 2D projections saved to: {output_path}")
    plt.close()






def create_3d_plot_html(directory):
    """
    Loads spatial_fingerprint_<identifier>.csv files from the directory and plots
    3D interactive point clouds with sample-wise hover info and visually correct legend.
    """

    pattern = re.compile(r"spatial_fingerprint_(.+?)\.csv")
    files = [
        file for file in os.listdir(directory)
        if pattern.match(file)
    ]

    if not files:
        print("No matching spatial fingerprint files found.")
        return

    # Prepare color palette
    color_palette = px.colors.qualitative.Plotly
    fig = go.Figure()

    for i, file in enumerate(sorted(files)):
        identifier = pattern.match(file).group(1)
        filepath = os.path.join(directory, file)
        color = color_palette[i % len(color_palette)]

        try:
            df = pd.read_csv(filepath, header=None)

            if df.shape[1] != 4:
                print(f"Skipping {file}: Expected 4 columns, got {df.shape[1]}")
                continue

            # Rename columns and clean up
            df.columns = ['sample_name', 'x', 'y', 'z']
            df = df.iloc[1:]  # Drop header row if needed
            df[['x', 'y', 'z']] = df[['x', 'y', 'z']].astype(float)

            # Actual data trace (not shown in legend)
            fig.add_trace(
                go.Scatter3d(
                    x=df['x'],
                    y=df['y'],
                    z=df['z'],
                    mode='markers',
                    name=None,
                    showlegend=False,
                    legendgroup=identifier,
                    marker=dict(size=2, color=color),
                    text=df['sample_name'],
                    hovertemplate='<b>%{text}</b><br>X=%{x:.2f}<br>Y=%{y:.2f}<br>Z=%{z:.2f}<extra></extra>'
                )
            )

            # Dummy trace for legend (larger symbol)
            fig.add_trace(
                go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode='markers',
                    name=identifier,
                    showlegend=True,
                    legendgroup=identifier,
                    marker=dict(size=12, color=color),
                    hoverinfo='skip'
                )
            )

        except Exception as e:
            print(f"Failed to load {file}: {e}")

    # Layout setup
    fig.update_layout(
        title="3D Spatial Fingerprint Point Clouds",
        scene=dict(
            xaxis_title="Noisy → Clustered Uncertainty (Moran)",
            yaxis_title="Structured → Diffuse Uncertainty (Entropy)",
            zaxis_title="Surface → Edge Uncertainty (EDS)"
        ),
        legend_title="Dataset",
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Save interactive plot
    output_path = os.path.join("output", "figures", "spatial_fingerprint_pointcloud.html")
    fig.write_html(output_path)
    print(f"Interactive 3D plot saved to: {output_path}. Open it in a browser to view.")



if __name__ == "__main__":
    create_3d_plot_html("output/tables")
    plot_2d_projections("output/tables")