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

FOLDER_PATH = '/fast/AG_Kainmueller/vguarin/aggrigator_experiments/output/tables/eaurc_id_ood/'
# NEW: specify the keyuq_methodword to search for in filenames
SEARCH_KEYWORD = 'tta'

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
    'Imbalance-w. class avg.': 'ICA', 
    'Equally-w. class avg.': 'BCA',
    'GMM_pixel': 'GMM-I', 
    'GMM_spatial': 'GMM-S', 
    'GMM': 'GMM-F',
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
search_pattern = os.path.join(FOLDER_PATH, '*_eaurc_id_ood_results.csv')
all_files = glob.glob(search_pattern)

# Filter by keyword (case-insensitive)
file_paths = [f for f in all_files if SEARCH_KEYWORD.lower() in os.path.basename(f).lower()]

all_means, all_stds = [], []
for filepath in file_paths:
    basename = os.path.basename(filepath)
    dataset_name = basename.replace('_eaurc_id_ood_results.csv', '')
    temp_df = pd.read_csv(filepath)
    temp_df['Aggregator'] = temp_df['Aggregator'].map(AGGREGATOR_NAME_MAPPING).fillna(temp_df['Aggregator'])
    temp_df = temp_df.set_index('Aggregator')
    all_means.append(temp_df[['EAURC']].rename(columns={'EAURC': dataset_name}))
    all_stds.append(temp_df[['EAURC_std']].rename(columns={'EAURC_std': dataset_name}))

summary_df_eaurc = pd.concat(all_means, axis=1)
stds_df_eaurc = pd.concat(all_stds, axis=1)
summary_df_eaurc = summary_df_eaurc.rename(columns=COLUMN_NAME_MAPPING)
stds_df_eaurc = stds_df_eaurc.rename(columns=COLUMN_NAME_MAPPING)

# --- Only for TTA and Ensemble, otherwise comment out !---
columns_to_keep = ['ARC-Nuc', 'LIDC-Tex', 'ARC-BC', 'LIDC-Mal', 'LIZ-IG', 'LIZ-SG'] #, 'CAR-CS'
summary_df_eaurc = summary_df_eaurc[columns_to_keep]
stds_df_eaurc = stds_df_eaurc[columns_to_keep]

# --- Calculate Mean Rank (where LOWEST is better) ---
ranks_df_eaurc = summary_df_eaurc.rank(ascending=True, method='min')
summary_df_eaurc['Mean Rank'] = ranks_df_eaurc.mean(axis=1)
final_column_order = [
    'ARC-BC', 'ARC-Nuc', 'LIDC-Mal', 'LIDC-Tex', #'CAR-CS',
    'LIZ-IG', 'LIZ-SG', 'Mean Rank'#'WEED-Hand', 'WORM-Nem', 'WORM-Pro', 'Mean Rank'
]

summary_df_eaurc = summary_df_eaurc[final_column_order]
summary_df_eaurc = summary_df_eaurc.sort_values(by='Mean Rank', ascending=True)
stds_df_eaurc = stds_df_eaurc.reindex(index=summary_df_eaurc.index, columns=summary_df_eaurc.columns.drop('Mean Rank', errors='ignore'))
summary_df_eaurc.index.name = None

# --- Create the Display DataFrame ---
display_df_eaurc = pd.DataFrame(index=summary_df_eaurc.index, columns=summary_df_eaurc.columns, dtype=str)
for col in display_df_eaurc.columns:
    if col == 'Mean Rank':
        # This formats the rank to one decimal place
        display_df_eaurc[col] = summary_df_eaurc[col].map('{:.1f}'.format)
    else:
        mean_series = summary_df_eaurc[col]
        std_series = stds_df_eaurc[col]
        display_df_eaurc[col] = mean_series.map('{:.3f}'.format) + ' ± ' + std_series.map('{:.3f}'.format)

# --- Final Styling ---
dataset_cols_eaurc = [col for col in summary_df_eaurc.columns if col != 'Mean Rank']

# --- THIS IS THE CORRECTED STYLING BLOCK ---
# Start with the base styler object from the correct display DataFrame
styled_df_eaurc = display_df_eaurc.style

# Loop through each data column and apply the gradient individually
for col in dataset_cols_eaurc:
    styled_df_eaurc = styled_df_eaurc.background_gradient(
        cmap=sns.diverging_palette(10, 130, as_cmap=True).reversed(),  # Reversed cmap for "lower is better"
        subset=[col],      # Apply to this specific column
        gmap=summary_df_eaurc[col], # Use the corresponding numeric column for color
        low=0.3,
        high=0.7
    )

# Chain the final properties after the loop
styled_df_eaurc = styled_df_eaurc.set_properties(**{'width': '100px'})


# print("\n--- Styled EAURC Summary Table (Lowest is Better, Column-Normalized) ---")
# display(styled_df_eaurc)


# --- Save final files ---
output_csv_path_eaurc = f'evaluation/scripts/{SEARCH_KEYWORD}_eaurc_summary_with_ranks.csv'
summary_df_eaurc.to_csv(output_csv_path_eaurc)
print(f"\nSuccessfully saved data to '{output_csv_path_eaurc}'")

# Export the final styled image
output_image_path_eaurc = f'evaluation/scripts/{SEARCH_KEYWORD}_eaurc_summary_table.png'
dfi.export(
    styled_df_eaurc,
    output_image_path_eaurc,
    table_conversion='matplotlib'
)
print(f"Successfully saved styled table as a PNG to '{output_image_path_eaurc}'.")