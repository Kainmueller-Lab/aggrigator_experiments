import nest_asyncio
nest_asyncio.apply()

import os
import glob
import pandas as pd
import seaborn as sns
import dataframe_image as dfi
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# --- Configuration ---

# The path to the folder the your CSV files.
FOLDER_PATH = '/fast/AG_Kainmueller/vguarin/aggrigator_experiments/output/tables/auroc_gmm/'
# NEW: specify the keyuq_methodword to search for in filenames
SEARCH_KEYWORD = 'ensemble'

# This dictionary maps the aggregator names in the CSV files
AGGREGATOR_NAME_MAPPING = {
    'Mean': 'AVG',
    'Quantile 0.6': 'AQA 0.60',
    'Quantile 0.75': 'AQA 0.75',
    'Quantile 0.9': 'AQA 0.90',
    'Patch 10': 'PLM 10',
    'Patch 20': 'PLM 20',
    'Patch 50': 'PLM 50',
    'Threshold 0.3': 'ATA 0.3',
    'Threshold 0.5': 'ATA 0.5',
    'Threshold 0.7': 'ATA 0.7',
    'Quantile fg. ratio': 'QFR',
    'Imbalance-w. class avg.': 'ICA', # Assuming this mapping
    'Equally-w. class avg.': 'BCA', # Assuming this mapping
    'GMM_pixel': 'GMM-Int',           # Assuming this mapping
    'GMM_spatial': 'GMM-Spa',         # Assuming this mapping
    'GMM': 'GMM-All',                 # Assuming this mapping
}

COLUMN_NAME_MAPPING = {
    f'instance_lizard_glas_set_{SEARCH_KEYWORD}_pu': 'LIZ-IG',
    f'fgbg_wormbodies_protists_{SEARCH_KEYWORD}_pu': 'WORM-Pro',
    f'instance_arctique_nuclei_intensity_{SEARCH_KEYWORD}_pu': 'ARC-Nuc',
    f'semantic_gta_cityscapes_{SEARCH_KEYWORD}_pu': 'CAR-CS',
    f'crops_vs_weed_weedsgalore_maize_{SEARCH_KEYWORD}_pu': 'WEED-Hand',
    f'fgbg_wormbodies_nematodes_{SEARCH_KEYWORD}_pu': 'WORM-Nem',
    f'fgbg_lidc_malignancy_{SEARCH_KEYWORD}_pu': 'LIDC-Mal',
    f'semantic_lizard_glas_set_{SEARCH_KEYWORD}_pu': 'LIZ-SG',
    f'fgbg_lidc_texture_{SEARCH_KEYWORD}_pu': 'LIDC-Tex',
    f'semantic_arctique_blood_cells_{SEARCH_KEYWORD}_pu': 'ARC-BC'
}

# --- Data Loading and Processing ---

# Find all relevant CSV files in the specified folder
search_pattern = os.path.join(FOLDER_PATH, '*_auroc_ood_results.csv')
all_files = glob.glob(search_pattern)

# Filter by keyword (case-insensitive)
file_paths = [f for f in all_files if SEARCH_KEYWORD.lower() in os.path.basename(f).lower()]

if not file_paths:
    print(f"Error: No CSV files found at '{search_pattern}'. Please check your FOLDER_PATH.")
else:
    print(f"Found {len(file_paths)} CSV files to process.")

all_means = []
all_stds = []

for filepath in file_paths:
    # Extract the dataset name from the filename.
    basename = os.path.basename(filepath)
    dataset_name = basename.replace('_auroc_ood_results.csv', '')

    # Read AUROC and AUROC_std ---
    temp_df = pd.read_csv(filepath)
    temp_df['Aggregator'] = temp_df['Aggregator'].map(AGGREGATOR_NAME_MAPPING).fillna(temp_df['Aggregator'])
    temp_df = temp_df.set_index('Aggregator')

    # Create and append the means DataFrame
    mean_df = temp_df[['AUROC']].rename(columns={'AUROC': dataset_name})
    all_means.append(mean_df)
    
    std_df = temp_df[['AUROC_std']].rename(columns={'AUROC_std': dataset_name})
    all_stds.append(std_df)
    
# Numeric DataFrame for calculations and color mapping
summary_df = pd.concat(all_means, axis=1)
stds_df = pd.concat(all_stds, axis=1)

summary_df = summary_df.rename(columns=COLUMN_NAME_MAPPING)
stds_df = stds_df.rename(columns=COLUMN_NAME_MAPPING)

# --- Only for TTA and Ensemble, otherwise comment out !---
columns_to_keep = ['ARC-Nuc', 'LIDC-Tex', 'ARC-BC', 'LIDC-Mal', 'CAR-CS',]
summary_df = summary_df[columns_to_keep]

# --- Calculate Rank, Reorder, and Sort ---
ranks_df = summary_df.rank(ascending=False, method='min')
summary_df['Mean Rank'] = ranks_df.mean(axis=1)
final_column_order = [
    'ARC-BC', 'ARC-Nuc', 'CAR-CS', 'LIDC-Mal', 'LIDC-Tex', 'Mean Rank'  #'CAR-CS', 
    # 'LIZ-IG', 'LIZ-SG', 'Mean Rank' #'WEED-Hand', 'WORM-Nem', 'WORM-Pro', 'Mean Rank'
]
summary_df = summary_df[final_column_order]
summary_df = summary_df.sort_values(by='Mean Rank', ascending=True)

# Align stds_df to the final sorted summary_df
stds_df = stds_df.reindex(index=summary_df.index, columns=summary_df.columns.drop('Mean Rank', errors='ignore'))
summary_df.index.name = None

# Create a second DataFrame with the desired string formats for display
display_df = pd.DataFrame(index=summary_df.index, columns=summary_df.columns, dtype=str)
for col in display_df.columns:
    if col == 'Mean Rank':
        display_df[col] = summary_df[col].map('{:.1f}'.format)
    else:
        # Create "mean ± std" strings
        mean_series = summary_df[col]
        std_series = stds_df[col]
        display_df[col] = mean_series.map('{:.3f}'.format) + ' ± ' + std_series.map('{:.3f}'.format)

# --- Final Styling ---
dataset_cols = [col for col in summary_df.columns if col != 'Mean Rank']

# Apply heatmap styling similar to the example image
# Green for high values, red for low, white for middle
styled_df = display_df.style.background_gradient(
    cmap=sns.diverging_palette(10, 130, as_cmap=True), # Red to Green palette #'RdYlGn',
    gmap=summary_df[dataset_cols],
    axis=None, 
    low=0.3, # Adjust these to control the color intensity
    high=0.7
).set_properties(
    **{'width': '100px'}
)

# print("\n--- Styled Summary Table with Mean Rank ---")
# Display the styled DataFrame in the notebook
# display(styled_df)

# --- Save the DataFrame for Sharing ---

# You can save the data in several formats.

# a) Save the raw data (without styles) to a CSV file
output_csv_path = f'evaluation/scripts/{SEARCH_KEYWORD}_auroc_summary_with_ranks.csv'
summary_df.to_csv(output_csv_path)
print(f"\nSuccessfully saved data to '{output_csv_path}'")

# b) Export using the Matplotlib backend
output_image_path = f'evaluation/scripts/{SEARCH_KEYWORD}_auroc_summary_table.png'

dfi.export(
    styled_df,
    output_image_path,
    table_conversion='matplotlib' # Ensures we use the reliable backend
)

print(f"Successfully saved styled table as a PNG to '{output_image_path}' using the Matplotlib backend.")