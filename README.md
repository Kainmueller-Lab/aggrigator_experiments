# Aggrigator Experiments 🐊

**Aggrigator** is a lightweight and modular Python library for aggregating uncertainty in deep learning workflows, especially useful for tasks like segmentation or per-pixel analysis.

With an intuitive API and a suite of built-in strategies, Aggrigator lets you:
- Reduce pixel-wise uncertainty maps into scalar scores for ranking or evaluation.
- Apply patch-based, class-specific, or thresholded aggregation strategies.
- Integrate spatial correlation metrics like Moran’s I or Geary’s C.
- Compare strategies side-by-side with insightful summaries and plots.

Designed to be modular, explainable, and research-friendly.  
Use it out of the box, or extend it with your own aggregation logic!

Bla bla

This repository reproduces results related to our publication.
📖 For details, see the [Aggrigator source code](https://github.com/Kainmueller-Lab/aggrigator).

## Installation

To install the aggrigator, clone the repository, navigate inside the directory and run the following command:

```bash
pip install -e .
```

now you can import the library in your python code with:

```python
import aggrigator
```

## Testing

To run the tests locally, navigate inside the aggrigator directory and first install the dev dependencies if needed:

```bash
pip install .[dev]
```

and then run the tests with:

```bash
pytest tests
```

## Try it out yourself

Check out the interactive [example_notebook.ipynb](example_notebook.ipynb) to see **Aggrigator** in action.  
You’ll learn how to:

- ✅ Generate and visualize uncertainty maps.  
- ⚙️ Apply and compare aggregation strategies.  
- 🧠 Use class-aware masks for targeted aggregation.
