from typing import Dict, Union, Literal, Optional
from torch import Tensor
import torch
import torch.nn.functional as F
from torchmetrics.segmentation import DiceScore
import numpy as np

import numpy as np
from typing import Optional

def debug_dice_calculation(preds, targets, num_classes=19, ignore_index=255):
    """
    Debug the dice calculation to understand why scores are too high
    """
    preds = np.asarray(preds, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    
    print("=== DEBUGGING DICE CALCULATION ===")
    print(f"Predictions shape: {preds.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Num classes: {num_classes}")
    print(f"Ignore index: {ignore_index}")
    
    # Check unique values
    pred_unique = np.unique(preds)
    target_unique = np.unique(targets)
    
    print(f"\nUnique prediction values: {pred_unique}")
    print(f"Unique target values: {target_unique}")
    
    # Check if ignore_index is actually present
    ignore_in_preds = ignore_index in pred_unique
    ignore_in_targets = ignore_index in target_unique
    
    print(f"\nIgnore index {ignore_index} in predictions: {ignore_in_preds}")
    print(f"Ignore index {ignore_index} in targets: {ignore_in_targets}")
    
    # Count pixels per class
    print(f"\n=== PIXEL COUNTS ===")
    for cls in range(num_classes):
        pred_count = np.sum(preds == cls)
        target_count = np.sum(targets == cls)
        print(f"Class {cls}: Pred={pred_count}, Target={target_count}")
    
    if ignore_index is not None:
        ignore_count_pred = np.sum(preds == ignore_index)
        ignore_count_target = np.sum(targets == ignore_index)
        print(f"Ignore class {ignore_index}: Pred={ignore_count_pred}, Target={ignore_count_target}")
    
    # Check if valid_mask is working correctly
    if ignore_index is not None:
        valid_mask = (targets != ignore_index)
        total_pixels = targets.size
        valid_pixels = np.sum(valid_mask)
        ignored_pixels = total_pixels - valid_pixels
        
        print(f"\n=== MASKING ===")
        print(f"Total pixels: {total_pixels}")
        print(f"Valid pixels: {valid_pixels}")
        print(f"Ignored pixels: {ignored_pixels}")
        print(f"Ignored percentage: {ignored_pixels/total_pixels*100:.2f}%")
    
    # Calculate dice for each sample individually to see distribution
    batch_size = preds.shape[0]
    individual_dice = []
    
    for i in range(batch_size):
        pred_sample = preds[i]
        target_sample = targets[i]
        
        # Simple dice calculation for this sample
        dice_score = calculate_simple_dice(pred_sample, target_sample, num_classes, ignore_index)
        individual_dice.append(dice_score)
        
        if i < 3:  # Show details for first 3 samples
            print(f"\nSample {i} dice: {dice_score:.6f}")
    
    individual_dice = np.array(individual_dice)
    print(f"\n=== INDIVIDUAL DICE SCORES ===")
    print(f"Mean: {individual_dice.mean():.6f}")
    print(f"Std: {individual_dice.std():.6f}")
    print(f"Min: {individual_dice.min():.6f}")
    print(f"Max: {individual_dice.max():.6f}")
    print(f"Median: {np.median(individual_dice):.6f}")
    
    return individual_dice

def calculate_simple_dice(pred, target, num_classes, ignore_index=None):
    """
    Simple dice calculation for a single sample to debug
    """
    pred = np.asarray(pred, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    
    if ignore_index is not None:
        valid_mask = (target != ignore_index)
        pred_masked = pred[valid_mask]
        target_masked = target[valid_mask]
    else:
        pred_masked = pred.flatten()
        target_masked = target.flatten()
    
    if len(pred_masked) == 0:
        return 0.0
    
    # Calculate intersection and union
    intersection = np.sum(pred_masked == target_masked)
    total = len(pred_masked)
    
    # This is actually accuracy, not dice - but let's see what we get
    accuracy = intersection / total
    
    return accuracy

def compare_ignore_strategies(preds, targets, num_classes=19):
    """
    Compare different ignore_index strategies
    """
    print("=== COMPARING IGNORE STRATEGIES ===")
    
    preds = np.asarray(preds, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    
    strategies = [
        ("No ignore", None),
        ("Ignore 255", 255),
        ("Ignore class 18", 18),
        ("Ignore class 0", 0),
    ]
    
    for strategy_name, ignore_idx in strategies:
        try:
            dice_scores = dice_torchmetrics_aligned(
                preds, targets, num_classes=num_classes, 
                ignore_index=ignore_idx, average='micro'
            )
            mean_dice = dice_scores.mean()
            print(f"{strategy_name}: {mean_dice:.6f}")
        except Exception as e:
            print(f"{strategy_name}: ERROR - {e}")

def dice_torchmetrics_aligned(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    average: str = 'micro',
    ignore_index: Optional[int] = None,
    smooth: float = 1.0
) -> np.ndarray:
    """
    Computes the Dice coefficient in a way that closely matches torchmetrics.Dice.

    Args:
        preds (np.ndarray): Predictions from the model. Assumed to be integer labels.
                            Shape: (N, H, W).
        targets (np.ndarray): Ground truth labels. Shape: (N, H, W).
        num_classes (int): The number of classes in the dataset.
        average (str): Defines the reduction to apply across classes. One of
                       'micro', 'macro', or 'none'. Defaults to 'micro'.
        ignore_index (Optional[int]): Specifies a target class to ignore.
                                      Pixels with this target value are excluded
                                      from the calculation. Defaults to None.
        smooth (float): A smoothing factor added to the numerator and denominator
                        to handle empty classes and avoid division by zero.
                        Defaults to 1.0, matching torchmetrics.

    Returns:
        np.ndarray: The calculated Dice score.
    """
    def pad_batch_to_array(batch_list, pad_value=0):
        """Convert a list of arrays with different shapes to a padded numpy array."""
        if isinstance(batch_list, np.ndarray):
            return batch_list
        
        # Convert to list of numpy arrays if needed
        batch_arrays = [np.asarray(item) for item in batch_list]
        
        # Find maximum dimensions
        max_dims = []
        for dim_idx in range(len(batch_arrays[0].shape)):
            max_dim = max(arr.shape[dim_idx] for arr in batch_arrays)
            max_dims.append(max_dim)
        
        # Create padded batch
        padded_batch = []
        for arr in batch_arrays:
            # Calculate padding for each dimension
            padding = []
            for dim_idx in range(len(arr.shape)):
                pad_amount = max_dims[dim_idx] - arr.shape[dim_idx]
                # Pad at the end (you can change this to center padding if needed)
                padding.append((0, pad_amount))
            
            # Pad the array
            padded_arr = np.pad(arr, padding, mode='constant', constant_values=pad_value)
            padded_batch.append(padded_arr)
        
        return np.stack(padded_batch, axis=0)
    
    try:
        preds = np.asarray(preds).astype(np.int64)
        targets = np.asarray(targets).astype(np.int64)
    except:
        # Convert lists to padded numpy arrays
        print("Converting and padding batch...")
        preds = pad_batch_to_array(preds, pad_value=ignore_index)
        targets = pad_batch_to_array(targets, pad_value=ignore_index)
        print(f"After padding: preds {preds.shape}, targets {targets.shape}")

    if preds.shape != targets.shape:
        raise ValueError(f"preds and targets must have the same shape, got {preds.shape} and {targets.shape}")

    # --- Create a mask for valid pixels from the ORIGINAL targets ---
    if ignore_index is not None:
        valid_mask = (targets != ignore_index)
    else:
        valid_mask = np.ones_like(targets, dtype=bool)

    # --- **FIX**: Handle ignore_index AND clip all values to valid range ---
    # Create safe versions of targets and preds for the indexing step to prevent IndexError.
    # The actual ignored pixels will be zeroed out later by the `valid_mask`.
    targets_for_ohe = targets.copy()
    if ignore_index is not None:
        # Replace ignored values with a safe, valid index (e.g., 0).
        # This is just a placeholder to prevent the IndexError.
        targets_for_ohe[targets == ignore_index] = 0
    
    # Clip ALL values to ensure they're within valid range [0, num_classes-1]
    targets_for_ohe = np.clip(targets_for_ohe, 0, num_classes - 1)
    preds_for_ohe = np.clip(preds, 0, num_classes - 1)

    # --- One-Hot Encoding using the safe versions ---
    preds_one_hot = np.eye(num_classes, dtype=np.float32)[preds_for_ohe]
    targets_one_hot = np.eye(num_classes, dtype=np.float32)[targets_for_ohe]

    preds_one_hot = np.moveaxis(preds_one_hot, -1, 1)
    targets_one_hot = np.moveaxis(targets_one_hot, -1, 1)

    # --- Core Metric Calculation (TP, FP, FN) ---
    # Reduce over spatial dimensions (H, W) but keep batch and class dimensions
    reduce_axes = tuple(range(2, preds_one_hot.ndim))  # Only spatial dimensions

    # Apply the valid_mask to exclude ignored pixels from all counts.
    # The mask is broadcast from (N, H, W) to (N, 1, H, W) to match the one-hot tensors.
    masked_preds = preds_one_hot * valid_mask[:, np.newaxis, :, :]
    masked_targets = targets_one_hot * valid_mask[:, np.newaxis, :, :]

    # tp, fp, fn will have shape (N, num_classes) - per batch, per class
    tp = np.sum(masked_preds * masked_targets, axis=reduce_axes)
    fp = np.sum(masked_preds, axis=reduce_axes) - tp
    fn = np.sum(masked_targets, axis=reduce_axes) - tp

    # --- Averaging Logic ---
    if average == 'micro':
        # For micro averaging, sum across classes for each batch
        total_tp = np.sum(tp, axis=1)  # Shape: (N,)
        total_fp = np.sum(fp, axis=1)  # Shape: (N,)
        total_fn = np.sum(fn, axis=1)  # Shape: (N,)
        
        numerator = 2 * total_tp + smooth
        denominator = 2 * total_tp + total_fp + total_fn + smooth
        
        return numerator / denominator  # Shape: (N,)

    numerator = 2 * tp + smooth
    denominator = 2 * tp + fp + fn + smooth
    per_class_score = numerator / denominator  # Shape: (N, num_classes)
    
    class_mask = np.ones(num_classes, dtype=bool)
    if ignore_index is not None:
        # Ensure the ignored class itself isn't included in 'macro' or 'none' averages,
        # but only if it's a valid class index.
        if 0 <= ignore_index < num_classes:
             class_mask[ignore_index] = False

    if average == 'none':
        # Return per-class scores for each batch, shape: (N, num_valid_classes)
        return per_class_score[:, class_mask]
    
    if average == 'macro':
        # Average across valid classes for each batch, shape: (N,)
        return np.mean(per_class_score[:, class_mask], axis=1)

    raise ValueError(f"Unknown average type: {average}. Must be one of ['micro', 'macro', 'none']")

def dice_coefficient_torchmetrics(preds, targets, ignore_index=True, ignore_value=0, num_classes=2, smooth=1e-6):
    """Computes the Dice coefficient between predictions and targets using NumPy.
    Args:
        targets: ground truth masks/labels
        preds: Binarized predictions
        ignore_index: Whether to ignore a specific class/value
        ignore_value: The specific value/class to ignore (default: 255)
        num_classes: Number of classes in the dataset
        smooth: Small number to avoid division with 0 in denominator
    Returns:
        Dice value per scalar or per batch
    """
    def pad_batch_to_array(batch_list, pad_value=0):
        """Convert a list of arrays with different shapes to a padded numpy array."""
        if isinstance(batch_list, np.ndarray):
            return batch_list
        
        # Convert to list of numpy arrays if needed
        batch_arrays = [np.asarray(item) for item in batch_list]
        
        # Find maximum dimensions
        max_dims = []
        for dim_idx in range(len(batch_arrays[0].shape)):
            max_dim = max(arr.shape[dim_idx] for arr in batch_arrays)
            max_dims.append(max_dim)
        
        # Create padded batch
        padded_batch = []
        for arr in batch_arrays:
            # Calculate padding for each dimension
            padding = []
            for dim_idx in range(len(arr.shape)):
                pad_amount = max_dims[dim_idx] - arr.shape[dim_idx]
                # Pad at the end (you can change this to center padding if needed)
                padding.append((0, pad_amount))
            
            # Pad the array
            padded_arr = np.pad(arr, padding, mode='constant', constant_values=pad_value)
            padded_batch.append(padded_arr)
        
        return np.stack(padded_batch, axis=0)
    
    def ignore_class(preds: np.ndarray, target: np.ndarray, ignore_val: int) -> tuple[np.ndarray, np.ndarray]:
        """Ignore a specific class value."""
        if ignore_val == 0:
            # If ignoring background (class 0), remove first channel
            preds = preds[:, 1:] if preds.shape[1] > 1 else preds
            target = target[:, 1:] if target.shape[1] > 1 else target
        else:
            # If ignoring a specific value (like 255), mask it out
            # Create a mask for valid pixels (not equal to ignore_value)
            valid_mask = target != ignore_val
            # Apply mask to both predictions and targets
            preds = preds * valid_mask[:, np.newaxis, :, :]  # Broadcasting for channel dimension
            target = target * valid_mask
        return preds, target
    
    def safe_divide(
        num: np.ndarray,
        denom: np.ndarray,
        zero_division: Union[float, Literal["warn", "nan"]] = 0.0,
    ) -> np.ndarray:
        """Safe division with handling for zero denominators."""
        denom_safe = np.where(denom != 0, denom, np.nan)
        result = np.divide(num, denom_safe)
        result = np.nan_to_num(result, nan=zero_division)
        return result
    
    # Convert to numpy arrays and ensure correct types
    try:
        preds = np.asarray(preds)
        targets = np.asarray(targets).astype(np.int64)
    except:
        # Convert lists to padded numpy arrays
        print("Converting and padding batch...")
        preds = pad_batch_to_array(preds, pad_value=ignore_value)
        targets = pad_batch_to_array(targets, pad_value=ignore_value)
        print(f"After padding: preds {preds.shape}, targets {targets.shape}")
    
    # Store original targets for masking if needed
    original_targets = targets.copy()
    
    # Handle ignore_value before one-hot encoding
    if ignore_index and ignore_value != 0:
        # Replace ignore_value with a valid class (e.g., 0) for one-hot encoding
        targets = np.where(targets == ignore_value, 0, targets)
    
    # Ensure preds are integers and within valid range
    preds = preds.astype(np.int64)
    
    # Clip values to valid class range to prevent IndexError
    preds = np.clip(preds, 0, num_classes - 1)
    targets = np.clip(targets, 0, num_classes - 1)
    
    # One-hot encode
    preds = np.eye(num_classes)[preds]  # shape: (batch, H, W, C)
    targets = np.eye(num_classes)[targets]  # shape: (batch, H, W, C)
    preds = np.moveaxis(preds, -1, 1)  # (batch, C, H, W)
    targets = np.moveaxis(targets, -1, 1)  # (batch, C, H, W)
    
    # Apply ignore logic
    if ignore_index:
        if ignore_value == 0:
            # Original background ignoring logic
            preds, targets = ignore_class(preds, targets, ignore_value)
        else:
            # For ignore_value like 255, mask out those pixels
            valid_mask = (original_targets != ignore_value).astype(np.float32)
            # Apply mask to one-hot encoded tensors
            preds = preds * valid_mask[:, np.newaxis, :, :]
            targets = targets * valid_mask[:, np.newaxis, :, :]
    
    # Compute intersection and union
    reduce_axes = tuple(range(2, preds.ndim))
    intersection = np.sum(preds * targets, axis=reduce_axes)
    union = np.sum(preds, axis=reduce_axes) + np.sum(targets, axis=reduce_axes)
    
    # Handle empty masks
    if np.all(union < smooth):
        return np.array(1.0, dtype=np.float32)
    
    # Compute Dice score (micro average)
    numerator = 2.0 * np.sum(intersection, axis=-1)
    denominator = np.sum(union, axis=-1)
    dice_score = safe_divide(numerator, denominator, zero_division=np.nan)
    print(dice_score)
    return dice_score  # shape: (batch,)

def dice_coefficient_torchmetrics_tensor(preds, targets, ignore_index=True, num_classes=2, smooth=1e-6):
    """Computes the Dice coefficient between predictions and targets as in torchmetrics.functional"""
    
    def _ignore_background(preds: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        """Ignore the background class in the computation assuming it is the first, index 0."""
        preds = preds[:, 1:] if preds.shape[1] > 1 else preds
        target = target[:, 1:] if target.shape[1] > 1 else target
        return preds, target

    def _safe_divide(
        num: Tensor,
        denom: Tensor,
        zero_division: Union[float, Literal["warn", "nan"]] = 0.0,
    ) -> Tensor:
        """Safe division, by preventing division by zero."""
        num = num if num.is_floating_point() else num.float()
        denom = denom if denom.is_floating_point() else denom.float()
        if isinstance(zero_division, (float, int)) or zero_division == "warn":
            zero_division = 0.0 
            zero_division_tensor = torch.tensor(zero_division, dtype=num.dtype).to(num.device, non_blocking=True)
            return torch.where(denom != 0, num / denom, zero_division_tensor)
        return torch.true_divide(num, denom)
    
    # Assuming output_softmax has the shape (batch_size, num_classes, H, W) - TODO: what happens if the validation batch is > 1?
    preds = (preds > .5).float() if preds.shape[1] == 1 else torch.argmax(preds, dim=1)  # Shape: (batch_size, H, W)
    print(preds.shape, targets.shape)
    # preds = preds.view(-1) # # Flatten tensors; Shape: (H*W)
    # targets = targets.view(-1) # Shape: (H*W)
    
    # Create one-hot encoding of predictions and targets
    num_classes = num_classes 
    print(preds.max(), preds.min(), num_classes)
    preds = F.one_hot(preds.long(), num_classes).float().movedim(-1, 1) #Shape: (batch_size, num_classes, extra_dim, H, W)
    targets = F.one_hot(targets.long(), num_classes).float().movedim(-1, 1) #Shape: (batch_size, num_classes, extra_dim, H, W)
    
    # Exclude the class at ignore_index (i.e., drop that column)
    if ignore_index:
        preds, targets = _ignore_background(preds, targets)
    
    # Calculate Dice coefficient for each class
    reduce_axis = list(range(2, targets.ndim))
    intersection = (preds*targets).sum(dim=reduce_axis) #Shape: (batch_size, num_classes)
    union = preds.sum(dim=reduce_axis) + targets.sum(dim=reduce_axis) #Shape: (batch_size, num_classes)
    
    # Handle empty masks - if both prediction and target are empty, consider it perfect match
    if union:
        comparable = union[:,0].item() if union.ndim > 2 else union.item()
        if comparable < smooth:
            return torch.tensor(1.0).to(preds.device)
    else:
        return torch.tensor(0.0).to(preds.device)
    
    # "micro" average - code other aggregation strategies if necessary
    numerator = torch.sum(2.0 * intersection, dim=-1) #sum over batches 
    denominator = torch.sum(union, dim=-1) 
    dice_score_other = _safe_divide(numerator, denominator, zero_division="nan")
    # dice_score = (2. * intersection + smooth) / (union + smooth) #Shape: (1, C) or (1, C-1)
    return dice_score_other #dice_score.mean() # over classes

def dice_coefficient(pred, target):
    """Compute Dice coefficient between prediction and target"""
    smooth = 1e-5
    intersection = torch.sum(pred * target) # Intersection
    union = torch.sum(pred) + torch.sum(target)  # Union
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

def dice_coefficient_improved(pred, target, smooth=1.0):
    """Compute Dice coefficient calculation with better handling of edge cases"""
    # Flatten tensors
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = torch.sum(pred_flat * target_flat) # Intersection
    union = torch.sum(pred_flat) + torch.sum(target_flat) # Union
    
    # Handle empty masks - if both prediction and target are empty, consider it perfect match
    if union.item() < smooth:
        return torch.tensor(1.0).to(pred.device)
    # Calculate Dice with appropriate smoothing factor
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

def multi_class_dice(pred, target, num_classes, smooth=1.0):
    """Extension of dice_coefficient_improved to multi-class segmentation"""
    dice_scores = []
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        dice = dice_coefficient_improved(pred_cls, target_cls, smooth)
        dice_scores.append(dice.item())
    return dice_scores

def calculate_mean_dice_across_raters(
    output_softmax: torch.Tensor, ground_truth: torch.Tensor, num_classes: int=2,
) -> Dict:
    """Calculate mean Dice across batch and raters as seen in ValUEs https://arxiv.org/pdf/2401.08501"""
    batch_size, num_raters, H, W = ground_truth.shape
    # Assuming output_softmax has the shape (batch_size, num_classes, H, W)
    all_dice_scores = []

    for r in range(num_raters):
        # Get the ground truth for the current batch and rater (shape: (H, W))
        gt_seg = ground_truth[:, r].unsqueeze(0).type(torch.LongTensor)  # Shape: (1, 1, H, W) 
        # Get the prediction probabilities (shape: (num_classes, H, W))
        # pred_softmax = output_softmax[i]  # Shape: (num_classes, H, W)
        # Get the predicted class for each pixel using the maximum probability from softmax (argmax over classes)
        # pred = (pred_softmax > .5).float() if pred_softmax.shape[0] == 1 else torch.argmax(pred_softmax, dim=0).unsqueeze(0)  # Shape: (1, H, W) 
        # Compute the Dice score between the prediction and ground truth for the current rate    
        print("values dice", output_softmax.shape, gt_seg.shape)
        test_dice = dice_coefficient_torchmetrics(output_softmax, gt_seg, num_classes=num_classes, ignore_index=True) #  dice_coefficient_improved(pred, gt_seg)
        # TODO: what happens if the validation batch is > 1?
        all_dice_scores.append(test_dice.item())
    return np.mean(all_dice_scores) # Calculate mean Dice score across all batches and raters