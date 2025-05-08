# test_environment.py

import numpy as np

def test_numpy_version():
    assert int(np.__version__.split('.')[0]) >= 2, f"NumPy version too old: {np.__version__}"

def test_fd_shifts_import():
    from fd_shifts.analysis.metrics import StatsCache  # raises if broken

def test_aggrigator_import():
    import aggrigator  # raises if broken
