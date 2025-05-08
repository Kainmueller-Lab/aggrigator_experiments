import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

# Simulate data
num_models = 5
num_samples = 100
num_classes = 3

# Simulated predictions (list of [num_samples x num_classes] softmax outputs)
pred_list = [np.random.rand(num_samples, num_classes) for _ in range(num_models)]
pred_list = [x / x.sum(axis=1, keepdims=True) for x in pred_list]  # softmax normalization

# Simulated ground-truth (length = num_samples)
gt_list = np.random.randint(0, num_classes, size=num_samples)

# Dummy class names
class_names = {i: f"class_{i}" for i in range(num_classes)}

# Shared data structure
shared_data = {"task": "classification"}

# Dummy acc_score function
def acc_score(gt_list, pred_array, weights, class_keys, n_classes, task):
    # Weighted ensemble prediction
    ensemble_preds = np.tensordot(weights, pred_array, axes=(0, 0))
    predicted_classes = ensemble_preds.argmax(axis=1)
    acc = (predicted_classes == gt_list).astype(float)
    return acc  # shape: (num_samples,)

# Dummy StatsCache class
class StatsCache:
    def __init__(self, uncertainty, accuracy, resolution=10):
        self.aurc = np.mean(uncertainty)  # dummy AURC
        self.coverages = np.linspace(1, 0, resolution + 1)
        self.selective_risks = np.linspace(0, 1, resolution + 1)

# Constants
AURC_DISPLAY_SCALE = 100
num_aggr = 4  # number of aggregation strategies

# Preallocate results
aggr_results = {}
aggr_acc = np.zeros((num_samples, num_aggr))
aggr_unc_val = np.zeros((num_samples, num_aggr))
augrc_res = {
    'augrc_val': np.zeros(num_aggr),
    'coverages': np.zeros(num_samples + 1),
    'generalized_risks': np.zeros((num_samples + 1, num_aggr)),
}

# Simulated aggregation function
def simulate_aggr(idx):
    weights = np.random.dirichlet(np.ones(num_models))
    uncertainty = 1 - np.max(weights)  # dummy "uncertainty" based on weight spread
    return idx, np.full(num_samples, uncertainty), weights

# Run the aggregation loop with simulated data
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(simulate_aggr, i) for i in range(num_aggr)]
    for future in tqdm(futures, desc="Processing aggregation strategies"):
        idx, aggr_unc, weights = future.result()
        aggr_results[idx] = weights
        aggr_acc_val = acc_score(
            gt_list, np.stack(pred_list, axis=0), weights,
            list(class_names.keys()), len(class_names), shared_data['task']
        )
        valid_mask = np.isnan(aggr_acc_val)
        aggr_acc[:, idx] = np.where(valid_mask, 0, aggr_acc_val)
        aggr_unc_val[:, idx] = np.where(valid_mask, 0, aggr_unc)

        evaluator = StatsCache(-aggr_unc_val[:, idx], aggr_acc[:, idx], 10)
        augrc_res['augrc_val'][idx] = evaluator.aurc / AURC_DISPLAY_SCALE
        coverage = evaluator.coverages
        if coverage.shape[0] < len(pred_list) + 1:
            coverage = np.append(coverage, 0)
        augrc_res['coverages'] = coverage
        risks = evaluator.selective_risks
        # Make sure that risks has the same shape as the slice in generalized_risks
        if risks.shape[0] < 101:
            risks = np.pad(risks, (0, 101 - risks.shape[0]), 'constant', constant_values=0)
        # Now, assign it to the correct place in generalized_risks
        augrc_res['generalized_risks'][:, idx] = risks
        print(f"AURC[{idx}] =", evaluator.aurc / AURC_DISPLAY_SCALE)
