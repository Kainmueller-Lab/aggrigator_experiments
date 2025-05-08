# Aggrigator Experiments 🐊

**Aggrigator** is a lightweight and modular Python library for aggregating uncertainty in deep learning workflows, especially useful for tasks like segmentation or per-pixel analysis.

With an intuitive API and a suite of built-in strategies, Aggrigator lets you:
- Reduce pixel-wise uncertainty maps into scalar scores for ranking or evaluation.
- Apply patch-based, class-specific, or thresholded aggregation strategies.
- Integrate spatial correlation metrics like Moran’s I or Geary’s C.
- Compare strategies side-by-side with insightful summaries and plots.

Designed to be modular, explainable, and research-friendly.  
Use it out of the box, or extend it with your own aggregation logic!

This repository reproduces results related to our publication.<br>
📖 For details, see the [Aggrigator source code](https://github.com/Kainmueller-Lab/aggrigator).


## Pre-requisites

Setup the environment by running the following commands. Be careful to choose the right pytorch version for your installed CUDA Version.

```bash
micromamba env create -f environment.yml
micromamba activate aggr_experiments
```

```selective_risk_coverage_curve.py``` relies on the [```fd-shifts```]() repository, which has a dependency on numpy<2. However, Aggrigator requires numpy>2 for optimal functionality. To avoid dependency conflicts, we recommend the following:

1. Clone the ```fd-shifts``` repository after setting up your environment.<br>
2. Edit the ```pyproject.toml``` file to replace

```toml
"numpy>=1.22.2,<2.0.0"
```

with 

```toml
"numpy>=2.0.0"
```

3. Then, install ```fd-shifts``` from local cloning using, 

```bash
(aggr_experiments) pip install -e /path/to/local/fd-shifts
```

This modification is safe because the parts of fd-shifts used in this experiment are compatible with ```numpy>=2.0.0```.
