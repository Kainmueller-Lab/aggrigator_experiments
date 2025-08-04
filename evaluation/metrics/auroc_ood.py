import os
import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Any, Tuple, Optional
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from scipy.stats import wilcoxon
from itertools import combinations

from aggrigator.uncertainty_maps import UncertaintyMap

# ---- AUROC computation to assess an aggregator's ability to detect OoD images. ----

def _compute_auroc_bootstrap(
    uncertainty_maps: List[UncertaintyMap],
    gt_labels: np.ndarray,
    aggregation_method: Callable,
    params: Any,
    category: str,
    ignore_index: int,
    n_bootstraps: int = 1000,
    ) -> List[float]:
    """
    Compute AUROC to assess an aggregator's ability to detect OoD images, by defining:
    - True Positive Rate (TPR) as the proportion of correctly identified OoD images,
    i.e., the fraction of OoD samples whose aggregated uncertainty score exceeds a given threshold.
    - False Positive Rate (FPR) as the proportion of iD images incorrectly classified as OoD,
    i.e., the fraction of iD samples whose aggregated uncertainty score surpasses the same threshold.
    """
    n_samples = len(uncertainty_maps)
    bootstrapped_aurocs = []

    if category == 'Context-aware':
        agg_values = np.array([
            aggregation_method(umap, params, ignore_index) for umap in uncertainty_maps
        ])
    else:
        agg_values = np.array([
            aggregation_method(umap, params) for umap in uncertainty_maps
        ])

    if category == 'Threshold':
        agg_values = np.nan_to_num(agg_values, nan=0)
        mask = (agg_values == -1) | (agg_values == 0)
        agg_values[mask] = 0

    for _ in range(n_bootstraps):
        indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
        if len(np.unique(gt_labels[indices])) < 2:
            continue

        fpr, tpr, _ = roc_curve(gt_labels[indices], agg_values[indices])
        bootstrapped_aurocs.append(auc(fpr, tpr))
    return bootstrapped_aurocs, agg_values


def evaluate_aggregation_strategy(
    cached_maps: Dict,
    aggr_name: str,
    aggr_method: Callable,
    param: Any,
    category: str,
    ignore_index: int,
    n_bootstraps: int
) -> Tuple[List[Dict], Dict[str, List[float]]]:
    """
    Evaluate a single aggregation strategy. Since we run one UQ method at a time,
    we don't need to key by UQ method.
    """
    auroc_stats_results = []
    bootstrap_samples = {}
    
    uq_method = next(iter(cached_maps.keys()))
    
    uncertainty_maps = cached_maps[uq_method]['maps']
    gt_labels = cached_maps[uq_method]['gt_labels']
    
    auroc_values, agg_values = _compute_auroc_bootstrap(
        uncertainty_maps, gt_labels, aggr_method, param, category, ignore_index, n_bootstraps
    )

    if auroc_values:
        bootstrap_samples[aggr_name] = auroc_values
        auroc_stats_results.append({
            'Aggregator': aggr_name,
            'AUROC': np.mean(auroc_values),
            'AUROC_std': np.std(auroc_values),
        })
    else:
        print(f"Warning: Could not compute AUROC for {aggr_name}. Skipping.")
        return auroc_stats_results, bootstrap_samples, None

    return auroc_stats_results, bootstrap_samples, agg_values


def _perform_pairwise_wilcoxon_tests_on_aggregators(
    bootstrap_samples_by_agg: Dict[str, List[float]],
    noise_level: str
) -> pd.DataFrame:
    """
    Performs one-sided pairwise Wilcoxon signed-rank tests to compare
    all aggregation methods directly. Expects a flat dictionary.
    """
    p_value_results = []
    aggr_names = list(bootstrap_samples_by_agg.keys())

    for agg1_name, agg2_name in combinations(aggr_names, 2):
        # This now works because every value in the dict is a list of floats.
        samples1 = bootstrap_samples_by_agg[agg1_name]
        samples2 = bootstrap_samples_by_agg[agg2_name]

        min_len = min(len(samples1), len(samples2))
        if min_len == 0:
            continue

        # Test H1: agg1 > agg2
        _, p_greater = wilcoxon(samples1[:min_len], samples2[:min_len], alternative='greater', zero_method='zsplit')
        # Test H1: agg2 > agg1
        _, p_less = wilcoxon(samples1[:min_len], samples2[:min_len], alternative='less', zero_method='zsplit')

        p_value_results.append({
            'Noise_Level': noise_level,
            'Comparison': f'{agg1_name}_vs_{agg2_name}',
            f'p_value ({agg1_name} > {agg2_name})': p_greater,
            f'p_value ({agg2_name} > {agg1_name})': p_less,
        })

    return pd.DataFrame(p_value_results)


def _evaluate_standard_strategies(
    cached_maps: Dict,
    strategies: Dict,
    noise_level: str,
    ignore_index: int,
    n_bootstraps: int
) -> Tuple[List[Dict], Dict[str, List[float]], Dict[str, np.ndarray]]:
    """Evaluates all standard pixel-based aggregation strategies."""
    all_auroc_stats = []
    all_bootstrap_samples = {}
    all_agg_values = {}

    for category, methods in strategies.items():
        for aggr_name, (aggr_method, param) in methods.items():
            try:
                stats_results, bootstrap_samples, agg_values = evaluate_aggregation_strategy(
                    cached_maps, aggr_name, aggr_method, param, category, ignore_index, n_bootstraps
                )
                if stats_results:
                    all_auroc_stats.extend(stats_results)
                    all_bootstrap_samples.update(bootstrap_samples)
                    if agg_values is not None:
                        all_agg_values[aggr_name] = agg_values
            except Exception as e:
                print(f"Error processing method {aggr_name} for noise level {noise_level}: {e}")
                continue

    return all_auroc_stats, all_bootstrap_samples, all_agg_values


def _evaluate_gmm_strategy(
    cached_maps: Dict,
    noise_level: str,
    dataset_name: str,
    task: str,
    variation: str,
    decomp: str,
    n_bootstraps: int
) -> Optional[Tuple[List[Dict], Dict[str, List[float]]]]:
    """Loads, aligns, and evaluates the pre-computed GMM score with bootstrapping."""
    first_uq_method = next(iter(cached_maps.keys()))
    if 'sample_names' not in cached_maps[first_uq_method]:
        return None, None

    scores_filename = f"{task}_{dataset_name}_{variation}_{decomp}_scores_standardize.csv"
    scores_filepath = os.path.join(os.getcwd(), "spatial", "results", scores_filename)

    if not os.path.exists(scores_filepath):
        print("Warning: GMM scores file not found, skipping GMM AUROC calculation.")
        return None, None

    print("----Processing aggregator function: GMM Normalized Score, in Spatial category----")

    score_columns_to_merge = ['uq_map_name', 'gt_label', 'ood_score_normalized_all', 'ood_score_normalized_magnitude', 'ood_score_normalized_spatial']
    gmm_scores_df = pd.read_csv(scores_filepath)
    gmm_scores_df.rename(columns={'Unnamed: 0': 'uq_map_name', 'is_ood': 'gt_label'}, inplace=True)
    gmm_scores_df = gmm_scores_df.astype({'uq_map_name': str, 'gt_label': int})
    gmm_scores_to_merge = gmm_scores_df[score_columns_to_merge]

    alignment_df = pd.DataFrame({
        'uq_map_name': cached_maps[first_uq_method]['sample_names'],
        'gt_label': cached_maps[first_uq_method]['gt_labels']
    }).astype({'uq_map_name': str, 'gt_label': int})

    final_scores = pd.merge(
        alignment_df, gmm_scores_to_merge, on=['uq_map_name', 'gt_label'], how='left'
    )
    final_scores.rename(columns={
        'ood_score_normalized_all': 'gmm_score',
        'ood_score_normalized_magnitude': 'gmm_score_pix',
        'ood_score_normalized_spatial': 'gmm_score_spat'}, inplace=True
    )
    final_scores.dropna(subset=['gmm_score', 'gmm_score_pix', 'gmm_score_spat'], inplace=True)

    if final_scores.empty:
        return None, None

    gmm_results = []
    gmm_bootstrap_samples = {}
    gmm_types = ['GMM', 'GMM_pixel', 'GMM_spatial']
    gmm_score_columns = ['gmm_score', 'gmm_score_pix', 'gmm_score_spat']

    for gmm_label, gmm_col in zip(gmm_types, gmm_score_columns):
        gt_labels = final_scores['gt_label'].values
        gmm_values = final_scores[gmm_col].values
        n_samples = len(gt_labels)
        bootstrapped_aurocs = []

        for _ in range(n_bootstraps):
            indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
            if len(np.unique(gt_labels[indices])) < 2:
                continue
            fpr, tpr, _ = roc_curve(gt_labels[indices], gmm_values[indices])
            bootstrapped_aurocs.append(auc(fpr, tpr))

        if not bootstrapped_aurocs:
            print(f"Warning: Could not compute AUROC for {gmm_label}. Skipping.")
            continue
        
        # This already has the correct, flat structure
        gmm_bootstrap_samples[gmm_label] = bootstrapped_aurocs

        gmm_results.append({
            'Aggregator': gmm_label,
            'AUROC': np.mean(bootstrapped_aurocs),
            'AUROC_std': np.std(bootstrapped_aurocs),
        })

    return gmm_results, gmm_bootstrap_samples, final_scores

# --- Main Function ---

def evaluate_all_strategies(
    cached_maps: Dict,
    strategies: Dict,
    noise_level: str,
    ignore_index: int,
    dataset_name: str,
    task: str,
    variation: str,
    decomp: str,
    n_bootstraps: int,
    output_path : Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates all pixel-based and spatial aggregation strategies for a given noise level.
    """
    if not cached_maps:
        print(f"Warning: cached_maps dictionary is empty for noise level {noise_level}. No strategies will be evaluated.")
        return pd.DataFrame(), pd.DataFrame()

    # Step 1: Evaluate standard aggregators
    auroc_stats, bootstrap_samples, standard_agg_values = _evaluate_standard_strategies(
        cached_maps, strategies, noise_level, ignore_index, n_bootstraps
    )

    # Step 2: Evaluate the GMM spatial aggregator
    gmm_results, gmm_bootstrap_samples, gmm_scores_df = _evaluate_gmm_strategy(
        cached_maps, noise_level, dataset_name, task, variation, decomp, n_bootstraps
    )
    if gmm_results:
        auroc_stats.extend(gmm_results)
        if gmm_bootstrap_samples:
            bootstrap_samples.update(gmm_bootstrap_samples)
    
   # Step 3: Create the reproducibility DataFrame for this specific comparison
    repro_df = None
    first_uq_method = next(iter(cached_maps.keys()))
    sample_names = cached_maps[first_uq_method].get('sample_names', [])
    gt_labels = cached_maps[first_uq_method].get('gt_labels', np.array([]))

    if sample_names and gt_labels.size > 0 and standard_agg_values:
        repro_df = pd.DataFrame(standard_agg_values)
        converted_names = [str(name.item()) if hasattr(name, 'item') else name for name in sample_names]
        repro_df['uq_map_name'] = converted_names
        # repro_df['uq_map_name'] = sample_names
        repro_df['is_ood'] = gt_labels

        if gmm_scores_df is not None and not gmm_scores_df.empty:
            gmm_to_merge = gmm_scores_df[['uq_map_name', 'gmm_score', 'gmm_score_pix', 'gmm_score_spat']].copy()
            gmm_to_merge.rename(columns={
                'gmm_score': 'GMM', 'gmm_score_pix': 'GMM_pixel', 'gmm_score_spat': 'GMM_spatial'
            }, inplace=True)
            repro_df = pd.merge(repro_df, gmm_to_merge, on='uq_map_name', how='left')

    if not auroc_stats:
        return pd.DataFrame(), pd.DataFrame()

    # Step 4: Create the main results DataFrame
    auroc_df = pd.DataFrame(auroc_stats)
    auroc_df['Noise_Level'] = noise_level
    auroc_df = auroc_df.sort_values('AUROC', ascending=False).reset_index(drop=True)

    # Step 5: Perform pairwise Wilcoxon tests on the now-consistent bootstrap samples
    p_values_df = _perform_pairwise_wilcoxon_tests_on_aggregators(bootstrap_samples, noise_level)

    return auroc_df, p_values_df, repro_df