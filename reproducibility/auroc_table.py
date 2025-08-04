"""Generate AUROC summary table for reproducibility data.

This script walks over all CSV files located in
`output/tables/auroc_reproducibility_repo`, computes the AUROC of several
aggregation uncertainty metrics against the ground-truth `is_ood` labels and
writes a summary table (`auroc_table.csv`) with one row per dataset.

The dataset (row) names are mapped via a predefined NAME_MAPPING dictionary so
that the resulting table matches the desired shorthand notation.
"""
from __future__ import annotations

import os
import glob
from typing import List, Dict, Any

import pandas as pd
from sklearn.metrics import roc_curve, auc

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "output",
    "tables",
    "auroc_reproducibility_repo",
)

OUTPUT_DIR = os.path.dirname(__file__)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "auroc_table.csv")

AGG_COLUMNS: List[str] = [
    "BCA", "ICA", "AVG",
    "ATA 0.3", "ATA 0.5", "ATA 0.7",
    "AQA 0.60", "AQA 0.75", "AQA 0.90",
    "QFR", "PLM 10", "PLM 20", "PLM 50",
    "GMM-All", "GMM-Int", "GMM-Spa",
]

NAME_MAPPING: Dict[str, str] = {
    "instance_lizard_glas_set_pu": "LIZ-IG",
    "fgbg_wormbodies_protists_pu": "WORM-Pro",
    "instance_arctique_nuclei_intensity_pu": "ARC-Nuc",
    "semantic_gta_cityscapes_pu": "CAR-CS",
    "crops_vs_weed_weedsgalore_maize_pu": "WEED-Hand",
    "fgbg_wormbodies_nematodes_pu": "WORM-Nem",
    "fgbg_lidc_malignancy_pu": "LIDC-Mal",
    "semantic_lizard_glas_set_pu": "LIZ-SG",
    "fgbg_lidc_texture_pu": "LIDC-Tex",
    "semantic_arctique_blood_cells_pu": "ARC-BC",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def compute_auroc(gt_labels: pd.Series, scores: pd.Series) -> float:
    """Compute AUROC for the given ground-truth labels and scores.

    A dedicated helper is used mainly for clarity and to handle edge cases such
    as constant scores that would otherwise raise an exception inside
    ``roc_curve``.
    """
    # If the column contains only NaNs, return NaN straight away.
    if scores.isna().all():
        return float("nan")

    try:
        fpr, tpr, _ = roc_curve(gt_labels, scores)
        return float(auc(fpr, tpr))
    except ValueError:
        # This can occur when ``gt_labels`` contains only one class or when the
        # scores are constant. Falling back to NaN keeps the table shape while
        # signalling the invalid computation.
        return float("nan")


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> None:
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {DATA_DIR}.")

    rows: List[Dict[str, Any]] = []

    for csv_path in csv_files:
        basename = os.path.splitext(os.path.basename(csv_path))[0]
        dataset_name = NAME_MAPPING.get(basename, basename)

        df = pd.read_csv(csv_path)
        if "is_ood" not in df.columns:
            raise KeyError(f"Column 'is_ood' not found in {csv_path}.")
        gt = df["is_ood"].astype(int)

        row: Dict[str, Any] = {"Dataset": dataset_name}

        for col in AGG_COLUMNS:
            if col not in df.columns:
                row[col] = float("nan")
                continue
            row[col] = compute_auroc(gt, df[col])

        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_df = results_df.set_index("Dataset")
    # Sort columns in specified order (makes sure the CSV has predictable
    # column order even if some aggregations are missing in certain datasets).
    results_df = results_df[AGG_COLUMNS]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df.to_csv(OUTPUT_CSV)
    print(f"AUROC summary table written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
