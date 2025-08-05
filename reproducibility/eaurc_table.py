"""Generate eAURC (selective risk) summary table for reproducibility data.

For every dataset we combine
    * dice scores   : output/tables/eaurc_reproducibility_repo/<dataset>.csv
    * uncertainties : output/tables/auroc_reproducibility_repo/<dataset>.csv

The two CSV‐files are merged on the ``uq_map_name`` column (left join with the
*Dice* table as the driver because it can be a subset of the samples).  Column
names are disambiguated by adding the suffixes ``_dice`` and ``_unc``.

For every aggregator in ``AGG_COLUMNS`` we compute the *expected area under the
risk–coverage curve* (eAURC, aka selective risk) following the formula that is
used throughout the project:

>>> evaluator = StatsCache(-uncertainty, dice, 10)
>>> eaurc      = evaluator.eaurc / AURC_DISPLAY_SCALE

The resulting scores are collected into a single table with datasets as rows
(using the same ``NAME_MAPPING`` that was used for the AUROC table) and written
as ``reproducibility/eaurc_table.csv``.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Any

import numpy as np
import pandas as pd

# External metric helper (provided by fd_shifts)
from fd_shifts.analysis.metrics import StatsCache  # type: ignore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

DICE_DIR = os.path.join(
    PROJECT_ROOT, "output", "tables", "eaurc_reproducibility_repo"
)
UNC_DIR = os.path.join(
    PROJECT_ROOT, "output", "tables", "auroc_reproducibility_repo"
)

OUTPUT_DIR = os.path.dirname(__file__)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "eaurc_table.csv")

AURC_DISPLAY_SCALE = 1000

AGG_COLUMNS: List[str] = [
    "BCA", "ICA", "AVG",
    "ATA 0.3", "ATA 0.5", "ATA 0.7",
    "AQA 0.60", "AQA 0.75", "AQA 0.90",
    "QFR", "PLM 10", "PLM 20", "PLM 50",
    "GMM-All", "GMM-Int", "GMM-Spa",
]

# Same mapping as before so tables line-up
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

def compute_eaurc(unc: np.ndarray, dice: np.ndarray) -> float:
    """Compute eAURC (selective risk) for a single aggregator."""
    # Remove NaNs prior to computation to avoid issues inside StatsCache
    valid = ~(np.isnan(unc) | np.isnan(dice))
    if valid.sum() == 0:
        return float("nan")

    evaluator = StatsCache(-unc[valid], dice[valid], 10)
    return float(evaluator.eaurc / AURC_DISPLAY_SCALE)


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> None:
    dice_files = sorted(glob.glob(os.path.join(DICE_DIR, "*.csv")))
    if not dice_files:
        raise RuntimeError(f"No CSV files found in {DICE_DIR}.")

    rows: List[Dict[str, Any]] = []

    for dice_path in dice_files:
        base = os.path.splitext(os.path.basename(dice_path))[0]
        unc_path = os.path.join(UNC_DIR, f"{base}.csv")
        if not os.path.exists(unc_path):
            raise FileNotFoundError(
                f"Uncertainty file not found for dataset '{base}': {unc_path}"
            )

        # ------------------------------------------------------------------
        # Load & prepare dataframes
        # ------------------------------------------------------------------
        dice_df = pd.read_csv(dice_path)
        unc_df = pd.read_csv(unc_path)

        # Rename aggregator columns to add suffixes
        dice_df = dice_df.rename(columns={c: f"{c}_dice" for c in AGG_COLUMNS if c in dice_df.columns})
        unc_df = unc_df.rename(columns={c: f"{c}_unc" for c in AGG_COLUMNS if c in unc_df.columns})

        merged = dice_df.merge(unc_df, on="uq_map_name", how="left")

        # Prepare arrays for computation
        results: Dict[str, Any] = {"Dataset": NAME_MAPPING.get(base, base)}

        for agg in AGG_COLUMNS:
            dice_col = f"{agg}_dice"
            unc_col = f"{agg}_unc"

            if dice_col not in merged.columns or unc_col not in merged.columns:
                results[agg] = float("nan")
                continue

            eaurc_val = compute_eaurc(
                merged[unc_col].to_numpy(float),
                merged[dice_col].to_numpy(float),
            )
            results[agg] = eaurc_val

        rows.append(results)

    res_df = pd.DataFrame(rows).set_index("Dataset")
    res_df = res_df[AGG_COLUMNS]  # ensure column order

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    res_df.to_csv(OUTPUT_CSV)
    print(f"eAURC summary table written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
