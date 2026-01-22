import pandas as pd
import numpy as np
import toml
import os
import argparse
import json
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

# --- CORE UTILS ---

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")
    return toml.load(config_path)

def load_spatial_fingerprints(filepath):
    try:
        df = pd.read_csv(filepath, index_col=0)
        return df[['moran', 'entropy', 'eds']]
    except: return None

def load_magnitude_fingerprints(filepath):
    try:
        df = pd.read_csv(filepath, index_col='uq_map_name')
        numeric_df = df.select_dtypes(include=np.number)
        return numeric_df[[c for c in numeric_df.columns if not c.lower().startswith('gmm')]]
    except: return None

def get_or_create_split(id_df, split_dir, base_filename):
    # Simplified split logic
    base_filename = "_".join(base_filename.split("_")[:-2] + base_filename.split("_")[-1:])
    path = os.path.join(split_dir, f"{base_filename}_train_split.json")
    if os.path.exists(path):
        with open(path, 'r') as f: train_idx = json.load(f)
        if "arctique" in base_filename:
            test_idx = [i for i in id_df.index if i not in train_idx] 
        else: 
            test_idx = [i for i in id_df.index.astype(str) if i not in train_idx]
    else:
        indices = id_df.index.astype(str).to_list()
        train_idx, test_idx = train_test_split(indices, test_size=0.5, random_state=42)
    return train_idx, test_idx

# --- GMM ENGINE ---

def run_gmm_evaluation(train_df, test_df, ood_df):
    """Fits GMM and returns AUROC."""
    # Preprocessing
    scaler = StandardScaler()
    train_np = scaler.fit_transform(train_df)
    test_np = scaler.transform(test_df)
    ood_np = scaler.transform(ood_df)

    p, n = train_df.shape[1], len(train_df)
    
    # Logic: Ensemble for high-dim, Single for low-dim (BIC)
    if (p**2 / n) > 0.5:
        # Ensemble
        scores_list = []
        for i in range(5): # Reduced models for ablation speed
            indices = np.random.choice(len(train_np), size=len(train_np), replace=True)
            gmm = GaussianMixture(n_components=min(p+2, 5), random_state=i).fit(train_np[indices])
            s = gmm.score_samples(np.vstack([test_np, ood_np]))
            scores_list.append(s)
        final_scores = -np.mean(scores_list, axis=0)
    else:
        # Single GMM with BIC
        bics = [GaussianMixture(n_components=i, random_state=42).fit(train_np).bic(train_np) for i in range(1, 6)]
        best_n = np.argmin(bics) + 1
        gmm = GaussianMixture(n_components=best_n, random_state=42).fit(train_np)
        final_scores = -gmm.score_samples(np.vstack([test_np, ood_np]))

    y_true = np.array([0]*len(test_df) + [1]*len(ood_df))
    fpr, tpr, _ = roc_curve(y_true, final_scores)
    return auc(fpr, tpr)

# --- ABLATION LOGIC ---

def run_ablation_study(train_df, test_df, ood_df, res_dir, base_filename):
    print(f"\n>>> Starting Leave-One-Out Ablation for {base_filename}")
    
    ablation_dir = os.path.join(res_dir, 'leave-one-out-ablation')
    os.makedirs(ablation_dir, exist_ok=True)
    
    features = train_df.columns.tolist()
    results = []

    # 1. Baseline
    print("Evaluating Baseline (All Features)...")
    baseline = run_gmm_evaluation(train_df, test_df, ood_df)
    results.append({'feature': 'ALL_FEATURES', 'mode': 'baseline', 'auroc': baseline, 'delta': 0})

    # 2. Leave-One-Out (Removal)
    for f in features:
        print(f" - Removing: {f}")
        cols = [c for c in features if c != f]
        score = run_gmm_evaluation(train_df[cols], test_df[cols], ood_df[cols])
        results.append({'feature': f, 'mode': 'removal', 'auroc': score, 'delta': score - baseline})

    # 3. Standalone (Addition/Only One)
    for f in features:
        print(f" - Standalone: {f}")
        score = run_gmm_evaluation(train_df[[f]], test_df[[f]], ood_df[[f]])
        results.append({'feature': f, 'mode': 'standalone', 'auroc': score, 'delta': score - baseline})

    # Save
    out_df = pd.DataFrame(results)
    out_path = os.path.join(ablation_dir, f"{base_filename}_ablation.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Done. Saved to {out_path}")

# --- PIPELINE ---

def run_pipeline(paths, base_filename):
    # Load Data
    id_s = load_spatial_fingerprints(paths['id_spatial'])
    ood_s = load_spatial_fingerprints(paths['ood_spatial'])
    id_m = load_magnitude_fingerprints(paths['id_magnitude'])
    ood_m = load_magnitude_fingerprints(paths['ood_magnitude'])
    
    if 'gta' in paths['id_spatial']:
        id_s.index = [f"{int(i):05d}" for i in id_s.index]
        id_m.index = [f"{int(i):05d}" for i in id_m.index]
                
    if id_s is None or id_m is None: return

    # Fix Index Overlaps
    if id_s.index.equals(ood_s.index):
        ood_s.index = [f"{i}_ood" for i in ood_s.index]
    if id_m.index.equals(ood_m.index):
        ood_m.index = [f"{i}_ood" for i in ood_m.index]

    # Handle Splits
    split_dir = os.path.join(os.getcwd(), 'spatial', 'splits')
    os.makedirs(split_dir, exist_ok=True)
    train_idx, test_idx = get_or_create_split(id_s, split_dir, base_filename)

    # Combine All Features
    id_all = pd.concat([id_s, id_m], axis=1).dropna()
    # Filter by available split indices
    train_idx = [i for i in train_idx if i in id_all.index]
    test_idx = [i for i in test_idx if i in id_all.index]
    
    id_train = id_all.loc[train_idx]
    id_test = id_all.loc[test_idx]
    ood_all = pd.concat([ood_s, ood_m], axis=1).dropna()

    res_dir = os.path.join(os.getcwd(), 'spatial', 'results')
    run_ablation_study(id_train, id_test, ood_all, res_dir, base_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    ds_name = config['dataset']['dataset_name']
    uq = config['pm']['uq_method']
    
    # Templates for paths (Generic for LIDC/Arctique/Standard)
    templates = {
        'id_spatial': config['paths'].get('id_csv_path_spatial'),
        'ood_spatial': config['paths'].get('ood_csv_path_spatial'),
        'id_magnitude': config['paths'].get('id_csv_path_magnitude'),
        'ood_magnitude': config['paths'].get('ood_csv_path_magnitude')
    }

    # Dataset specific looping (Logic adapted from your snippet)
    if ds_name == 'lidc':
        for var in ['malignancy', 'texture']:
            curr = {k: v.replace(config['dataset']['variation'], var) for k, v in templates.items()}
            base = f"{config['dataset']['task']}_{ds_name}_{var}_{uq}_pu"
            run_pipeline(curr, base)
    elif ds_name == 'arctique':
        original_task = config['dataset']['task']
        original_variation = config['dataset']['variation']
        original_noise = config['dataset']['noise_level']
        for task, var, ns in zip(['semantic', 'instance'], ['blood_cells', 'nuclei_intensity'], ['0_75', '0_50']):
            print(f"\n\n{'='*25} PROCESSING TASK: {task.upper()}; VARIATION: {var.upper()}  {'='*25}")
            curr = {
                k: (
                    v.replace(original_task, task)
                    .replace(original_variation, var)
                    .replace(original_noise, ns) if k.startswith('ood')
                    else v.replace(original_task, task).replace(original_variation, var)
                )
                for k, v in templates.items()
            }
            base = f"{task}_{ds_name}_{var}_{uq}_pu"
            run_pipeline(curr, base)
    elif ds_name == 'lizard':
        original_task = config['dataset']['task']
        original_variation = config['dataset']['variation']
        for task in ['semantic', 'instance']:
            print(f"\n\n{'='*25} PROCESSING TASK: {task.upper()}")
            curr = {k: v.replace(original_task, task) for k, v in templates.items()}
            base = f"{task}_{ds_name}_{original_variation}_{uq}_pu"
            run_pipeline(curr, base)
    elif ds_name == 'wormbodies':
        original_task = config['dataset']['task']
        original_variation = config['dataset']['variation']
        for var in ['nematodes', 'protists']:
            print(f"\n\n{'='*25} PROCESSING VARIATION: {var.upper()} {'='*25}")
            curr = {
                k: (
                    v.replace(original_variation, var) if not (k.startswith('id') and k.endswith('spatial')) else v
                )
                for k, v in templates.items()
            }
            base = f"{original_task}_{ds_name}_{var}_{uq}_pu"
            run_pipeline(curr, base)
    else:
        # Standard
        variation = config['dataset']['variation']
        base = f"{config['dataset']['task']}_{ds_name}_{variation}_{uq}_pu" if variation else  f"{config['dataset']['task']}_{ds_name}_{uq}_pu"
        run_pipeline(templates, base)