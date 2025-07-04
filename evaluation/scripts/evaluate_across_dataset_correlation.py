import glob
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compute_and_save_correlation(df, correlation_method, uq_method):
    # Drop non-numeric identifier columns
    columns_to_exclude = ['uq_map_name', 'dataset_name']
    columns_to_use = [col for col in df.columns if col not in columns_to_exclude]

    # Compute correlation matrix
    corr_matrix = df[columns_to_use].corr(method=correlation_method)

    filename = f"correlation_matrix_{correlation_method}_all_datasets_{uq_method}_pu"

    # Save to csv
    corr_matrix.to_csv(os.path.join("output", "tables", f"{filename}.csv"))
    print(f"Correlation matrix {filename}.csv saved to output folder.")

    # Save as heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    strategy_names = columns_to_use

    sns.heatmap(corr_matrix, ax=ax, cmap="coolwarm", annot=False, fmt=".2f",
                cbar=True, vmin=-1, vmax=1, xticklabels=strategy_names, yticklabels=strategy_names)
    
    # Color strategy names by category
    color_code = {
        "threshold": "red",
        "quantile": "green",
        "patch": "blue",
        "class_mean": "orange",
    }
    for tick in ax.get_xticklabels():
        strategy_name = tick.get_text()
        color = next((color_code[key] for key in color_code if key in strategy_name), "black")
        tick.set_bbox(dict(facecolor=color, edgecolor='none', alpha=0.5, boxstyle="round,pad=0.3"))
    for tick in ax.get_yticklabels():
        strategy_name = tick.get_text()
        color = next((color_code[key] for key in color_code if key in strategy_name), "black")
        tick.set_bbox(dict(facecolor=color, edgecolor='none', alpha=0.5, boxstyle="round,pad=0.3"))

    plt.title(filename)
    plt.savefig(os.path.join("output", "figures", f"{filename}.png"))
    plt.close()
    print(f"Correlation heatmap {filename}.png saved to output folder.")


def main():
    for uq_method in ["dropout", "softmax"]:
        # Read and concatenate all matching CSV files
        all_dfs = []
        for file_path in glob.glob(os.path.join("output", "tables", "aggregation_value_summary_*.csv")):
            if uq_method in file_path:
                df = pd.read_csv(file_path)
                all_dfs.append(df)

        # Stack all dataframes vertically
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Save to csv
        combined_df.to_csv(os.path.join("output", "tables", "aggregation_value_summary_all_datasets.csv"))

        for correlation_method in ["pearson", "spearman", "kendall"]:
            compute_and_save_correlation(combined_df, correlation_method, uq_method)


if __name__ == "__main__":
    main()
