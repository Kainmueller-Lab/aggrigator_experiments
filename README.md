# Aggrigator Experiments 🐊 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Aggrigator** is a lightweight and modular Python library for aggregating uncertainty in deep-learning workflows, especially useful for tasks like segmentation or other per-pixel analyses.

**This repository** accompanies the paper *“Better than Average: Shedding Light on the Impact of Aggregation Strategies for Segmentation Uncertainty.”*

---

## Installation

Create and activate the experiment environment (tested with **Python ≥ 3.10** and **CUDA 11.8 / 12.1**):

```bash
micromamba env create -f environment.yml
micromamba activate aggr_experiments
```

> The full dependency lists live in [`pyproject.toml`](pyproject.toml) (run-time) and [`environment.yml`](environment.yml) (conda-solvable build).

`evaluation/scripts/evaluate_aurc.py` depends on the external **[fd-shifts](https://github.com/IML-DKFZ/fd-shifts)** repository which originally pins `numpy<2.0.0`.  Aggrigator works with `numpy ≥ 2.0.0`; therefore:

1. Clone **fd-shifts** after creating the environment.
2. In its `pyproject.toml`, replace `"numpy>=1.22.2,<2.0.0"` with `"numpy>=2.0.0"`.
3. Install the modified package:

```bash
(aggr_experiments) pip install -e /path/to/local/fd-shifts
```

---

## Repository Map

| Path | Purpose |
|------|---------|
| `datasets/` | Download, convert, and standardzse the raw data used in the experiments (`*_dataset_creation.py`). |
| `evaluation/` | Metric implementations (AUROC, AURC, E_AURC, Dice, etc.) and high-level evaluation scripts for comprehensive benchmarking. |
| `spatial/` | Code for the spatial fingerprint approach, including spatial mass ratio computation, Gaussian Mixture Model (GMM) fitting, and feature preprocessing ablations. |
| `reproducibility/` | Lightweight scripts to reproduce key plots from the paper (e.g., Figure 4 and Figure 5b). |
| `output/` | Generated figures (`.png`, `.html`) and result tables in CSV format. |
| `tests/` | Unit tests (using PyTest) to ensure the reliability of core functionalities. |
| `environment.yml / pyproject.toml` | Environment and dependency specifications using Conda & Poetry. |

---

## Environment Variables (`.env`)

Some dataset creation and spatial scripts expect the variable `LAB_PATH` to point to your internal data storage. Create a `.env` file in the repo root (or export the variable in your shell):

```dotenv
# .env example
LAB_PATH=/absolute/path/to/your/lab/storage
```

Load it via `source .env` or let tools such as *direnv* / *dotenv-cli* handle it automatically.

---

## Evaluation
The repository addresses five practical questions:

1. **Similarity of uncertainty scores** – see `experiments/correlation_analyses/*.ipynb`.
2. **Reliability in real-world scenarios** – run `evaluation/scripts/evaluate_auroc.py` and `evaluation/scripts/evaluate_aurc.py`.
3. **Influence of parameters in non-parameter-free aggregators** – *coming soon*.
4. **Effect on model ranking in benchmarks** – *coming soon*.
5. **Spatial measures as context-aware aggregators** – see `evaluation/analyse_spatial_methods.py`.

---

## Reproducing the Main Results

The scripts below recreate Figures 4 & 5 of the paper and write tables/plots to `reproducibility/`:

```bash
python reproducibility/auroc_table.py  --save_dir reproducibility
python reproducibility/eaurc_table.py  --save_dir reproducibility
```

---

## Citation
If you use this code, please cite our work:

```BibTeX
@inproceedings{anon2024better,
  title        = {Better than Average: Shedding Light on the Impact of Aggregation Strategies for Segmentation Uncertainty},
  author       = {Anonymous},
  year         = {2025}
}
```

---

## License

This project is licensed under the **MIT License**.  See the [LICENSE](LICENSE) file for details.
