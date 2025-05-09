import numpy as np
import concurrent.futures
import math
from concurrent.futures import ProcessPoolExecutor
from skimage.measure import regionprops, label
from scipy.optimize import linear_sum_assignment

def get_class(regprop, cls_map):
    return cls_map[
        regprop.bbox[0] : regprop.bbox[2], regprop.bbox[1] : regprop.bbox[3]
    ][regprop.image][0]
    
def euclidean_dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def expand_bbox(bb, size_px=3, shape_constraints=None):
    """Expand bounding box by size_px in all directions."""
    min_row, min_col, max_row, max_col = bb
    min_row = max(min_row - size_px, 0)
    min_col = max(min_col - size_px, 0)
    max_row = (
        min(max_row + size_px, shape_constraints[0] - 1)
        if shape_constraints
        else max_row + size_px
    )
    max_col = (
        min(max_col + size_px, shape_constraints[1] - 1)
        if shape_constraints
        else max_col + size_px
    )
    return min_row, min_col, max_row, max_col

def per_tile_worker(cnt, gt_tile, pred_tile, match_euc_dist, class_names):
    gt_inst = label(gt_tile[..., 0], connectivity=2)
    pred_inst = label(pred_tile[..., 0], connectivity=2)
        
    true_id_list = np.arange(np.max(gt_inst))
    pred_id_list = np.arange(np.max(pred_inst))
    pairwise_cent_dist = np.full(
        [len(true_id_list), len(pred_id_list)], 1000, dtype=np.float64
    )
    
    true_objects = regionprops(gt_inst)
    pred_objects = regionprops(pred_inst)
    
    for ti, o in enumerate(true_objects):
        bb = expand_bbox(o.bbox, size_px=2, shape_constraints=gt_inst.shape)
        # gt_mask = gt[c,...,0][bb[0]:bb[2], bb[1]:bb[3]]==o.label
        valid = np.unique(pred_inst[bb[0] : bb[2], bb[1] : bb[3]])
        for pred_id in valid:
            if pred_id == 0:
                continue
            pred_obj = pred_objects[pred_id - 1]
            pairwise_cent_dist[ti, pred_id - 1] = euclidean_dist(
                o.centroid, pred_obj.centroid
            )
    paired_true, paired_pred = linear_sum_assignment(pairwise_cent_dist) #Hungarian algorithm matching predicted instances to ground truth
    paired_cen = pairwise_cent_dist[paired_true, paired_pred]
    paired_true = list(paired_true[paired_cen < match_euc_dist] + 1) #to determine whether a matched pair of ground truth and predicted instances should be considered a true positive 
    #Only pairs whose centroid distance is less than match_euc_dist are considered valid matches (true positives)
    
    paired_pred = list(paired_pred[paired_cen < match_euc_dist] + 1)
    paired_cen = paired_cen[paired_cen < match_euc_dist]

    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    
    try:
        f1 = (2 * tp) / ((2 * tp) + fp + fn)
    except ZeroDivisionError:
        # this means neither on GT nor pred there is a nucleus
        f1 = np.nan
    precision = np.float64(tp) / np.maximum(tp + fp, 1)
    recall = np.float64(tp) / np.maximum(tp + fn, 1)
    ap = precision * recall if tp > 0 else 0.0
    
    class_pairs = []
    
    for i, j in zip(paired_true, paired_pred):
        oi = true_objects[i - 1]
        oj = pred_objects[j - 1]
        oi_class = get_class(oi, gt_tile[..., 1])
        oj_class = get_class(oj, pred_tile[..., 1])
        class_pairs.append((oi_class, oj_class))
    
    class_pairs = np.array(class_pairs)
    
    add_fp = []
    for i in unpaired_pred:
        o = pred_objects[i - 1]
        o_class = get_class(o, pred_tile[..., 1])
        add_fp.append(o_class)
    add_fp = np.array(add_fp)
    add_fn = []
    for i in unpaired_true:
        o = true_objects[i - 1]
        o_class = get_class(o, gt_tile[..., 1])
        add_fn.append(o_class)
    add_fn = np.array(add_fn)
    
    sub_metrics = []
    for cls in np.arange(1, len(class_names) + 1):
        try:
            t = class_pairs[:, 0] == cls
            p = class_pairs[:, 1] == cls
            t_n = class_pairs[:, 0] != cls
            p_n = class_pairs[:, 1] != cls

            tp_c = np.count_nonzero(t & p)

            fp_c = np.count_nonzero(t_n & p) + np.count_nonzero(add_fp == cls)
            fn_c = np.count_nonzero(t & p_n) + np.count_nonzero(add_fn == cls)
            tn_c = np.count_nonzero(t_n & p_n)
        except IndexError:
            # fix no match for any class
            tp_c = 0
            fp_c = np.count_nonzero(add_fp == cls)
            fn_c = np.count_nonzero(add_fn == cls)
            tn_c = 0
        try:
            f1_c = (2 * tp_c) / ((2 * tp_c) + fp_c + fn_c)
        except ZeroDivisionError:
            f1_c = np.nan
        try:
            precision_c = 1. * (tp_c) / max(1, tp_c + fp_c)
            recall_c = 1. * (tp_c) / max(1, tp_c + fn_c)
            ap_c = precision_c * recall_c
        except ZeroDivisionError:
            ap_c = np.nan
            
        sub_metrics.append(
            {
                "id": cnt,
                "class": class_names[cls - 1],
                "TP": tp_c,
                "FP": fp_c,
                "FN": fn_c,
                "TN": tn_c,
                "F1": f1_c,
                "AP" : ap_c,
            }
        )

    sub_metrics.append(
        {
            "id": cnt,
            "class": "all",
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "F1": f1,
            "AP" : ap,
        }
    )
    return sub_metrics
    
def per_tile_metrics(gt, pred, class_names, match_euc_dist=6):
    metrics = []
    with ProcessPoolExecutor(4) as executor:
        future_metrics = []
        for cnt, (gt_tile, pred_tile) in enumerate(zip(gt, pred)):
            future_metrics.append(
                executor.submit(
                    per_tile_worker,
                    cnt,
                    gt_tile,
                    pred_tile,
                    match_euc_dist,
                    class_names,
                )
            )
        res = [
            future.result()
            for future in concurrent.futures.as_completed(future_metrics)
        ]
    res = [i for i in sorted(res, key=lambda x: x[0]["id"])]
    for i in res:
        metrics.extend(i)
    return metrics