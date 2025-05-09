# Aggrigator Experiments 🐊

**Aggrigator** is a lightweight and modular Python library for aggregating uncertainty in deep learning workflows, especially useful for tasks like segmentation or per-pixel analysis.

With an intuitive API and a suite of built-in strategies, Aggrigator lets you:
- Reduce pixel-wise uncertainty maps into scalar scores for ranking or evaluation.
- Apply patch-based, class-specific, or thresholded aggregation strategies.
- Integrate spatial correlation metrics like Moran’s I or Geary’s C.
- Compare strategies side-by-side with insightful summaries and plots.

Designed to be modular, explainable, and research-friendly, this repository also includes the code used to reproduce the results presented in the original publication introducing the library. <br>

📖 For full documentation and contribution guidelines, see the [Aggrigator source code](https://github.com/Kainmueller-Lab/aggrigator) and and open an issue or pull request to get involved!


## Prerequisites

Setup the environment by running the following commands. Be careful to choose the right pytorch version for your installed CUDA Version.

```bash
micromamba env create -f environment.yml
micromamba activate aggr_experiments
```

```selective_risk_coverage_curve.py``` relies on the [```fd-shifts```](https://github.com/IML-DKFZ/fd-shifts/tree/main) repository, which has a dependency on ```"numpy<2.0.0"```. However, Aggrigator requires ```"numpy>2.0.0"``` for optimal functionality. To avoid dependency conflicts, we recommend the following:

1. Clone the ```fd-shifts``` repository after setting up your environment.
2. Edit the ```pyproject.toml``` file to replace ```"numpy>=1.22.2,<2.0.0"``` with ```"numpy>=2.0.0"```.
3. Then, install ```fd-shifts``` from local cloning using

```bash
(aggr_experiments) pip install -e /path-to-local/fd-shifts
```

This modification is safe because the functions of ```fd-shifts``` used in this experiment are compatible with ```numpy>=2.0.0```. Once everything is installed and configured correctly, run the test suite to make sure all components work as expected:

```bash
(aggr_experiments) pytest -v
```

## Evaluation
To quantify the impact of choosing an aggregation method in one's use case, the repo offers answers to the following five questions:

1. How similar are the aggregated uncertainty scores produced by the different aggregators? <br> cf. ```experiments/correlation_analyses/*.ipynb```
2. When translating a UQ method into a real-world scenario, how does the aggregator affect its reliability? <br> cf. ```evaluation/scripts/evaluate_auroc.py```
3. To what extent does parameter choice in non–parameter-free aggregators modify method reliability? <br> cf. 
4. How can an aggregator impact on the selection of an optimal UQ model in benchmarking environments? <br> cf. 
5. How can spatial measures improve the aggregation performance of context-free aggregators? <br> cf. ```evaluation/analyse_spatial_methods.py```

