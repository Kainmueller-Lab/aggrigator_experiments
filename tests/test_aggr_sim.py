import numpy as np
from fd_shifts.analysis.metrics import StatsCache

AURC_DISPLAY_SCALE = 1000

def test_simulated_aggregation():
    num_samples = 100
    num_aggr = 4

    aggr_unc_val = np.random.rand(num_samples, num_aggr)        # Values in [0, 1]
    aggr_acc = np.random.rand(num_samples, num_aggr)            # Values in [0, 1]                      

    for idx in range(num_aggr):
        uncertainty = 1-aggr_unc_val[:, idx] # Treat lower uncertainty as higher confidence
        accuracy = aggr_acc[:, idx]
        evaluator = StatsCache(uncertainty, accuracy, num_aggr-1)
        assert 0 <= evaluator.aurc/AURC_DISPLAY_SCALE <= 1, f"Invalid AURC: {evaluator.aurc/AURC_DISPLAY_SCALE}"
