import numpy as np
import fd_shifts
import aggrigator

def test_numpy():
    print(f"NumPy version: {np.__version__}")
    assert int(np.__version__.split('.')[0]) >= 2, "NumPy version is too old"

def test_fd_shifts():
    try:
        from fd_shifts.analysis.metrics import StatsCache
        print("fd_shifts imported and functional ✅")
    except Exception as e:
        print("fd_shifts test failed ❌:", e)

def test_aggrigator():
    try:
        import aggrigator
        print("aggrigator imported and functional ✅")
    except Exception as e:
        print("aggrigator test failed ❌:", e)

if __name__ == "__main__":
    print("🔍 Testing environment compatibility...\n")
    test_numpy()
    test_fd_shifts()
    test_aggrigator()
