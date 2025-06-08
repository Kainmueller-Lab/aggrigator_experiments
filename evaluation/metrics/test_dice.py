import numpy as np
from typing import Union, Literal
from dice import *

# Alternative implementation with per-class dice scores
def continuous_dice_per_class(preds, targets, ignore_index=True, num_classes=2, smooth=1e-6):
    """Same as above but returns per-class Dice scores instead of averaged."""
    
    def ignore_background(preds: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ignore the background class (assumed index 0)."""
        preds = preds[:, 1:] if preds.shape[1] > 1 else preds
        target = target[:, 1:] if target.shape[1] > 1 else target
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
    
    def softmax(x, axis=1):
        """Compute softmax along specified axis."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    # Convert to numpy arrays
    preds = np.asarray(preds, dtype=np.float32)
    targets = np.asarray(targets)
    
    # Handle different input formats
    if preds.ndim == 4:
        if preds.shape[-1] == num_classes and preds.shape[1] != num_classes:
            preds = np.moveaxis(preds, -1, 1)
    
    # Apply softmax if needed
    if len(preds.shape) == 4:
        prob_sums = np.sum(preds, axis=1)
        if np.any(preds < 0) or np.any(preds > 1) or not np.allclose(prob_sums, 1.0, atol=0.1):
            preds = softmax(preds, axis=1)
    
    # Handle targets
    if targets.ndim == 3:
        targets = targets.astype(np.int64)
        targets = np.clip(targets, 0, num_classes - 1)
        targets_onehot = np.eye(num_classes)[targets]
        targets = np.moveaxis(targets_onehot, -1, 1)
    elif targets.ndim == 4:
        if targets.shape[-1] == num_classes and targets.shape[1] != num_classes:
            targets = np.moveaxis(targets, -1, 1)
        targets = targets.astype(np.float32)
    
    # Optionally ignore background
    if ignore_index and preds.shape[1] > 1:
        preds, targets = ignore_background(preds, targets)
    
    # Compute per-class Dice with c factor
    reduce_axes = tuple(range(2, preds.ndim))
    intersection = preds * targets  # Element-wise product
    intersection_sum = np.sum(intersection, axis=reduce_axes)
    
    # Compute c factor for each batch and class
    c_values = np.zeros_like(intersection_sum)
    for b in range(preds.shape[0]):
        for cls in range(preds.shape[1]):
            intersect_region = intersection[b, cls]
            positive_count = np.sum(intersect_region > 0)
            if positive_count > 0:
                c_values[b, cls] = np.sum(intersect_region) / positive_count
            else:
                c_values[b, cls] = 1.0
    
    pred_sum = np.sum(preds, axis=reduce_axes)
    target_sum = np.sum(targets, axis=reduce_axes)
    
    numerator = 2.0 * intersection_sum
    denominator = c_values * target_sum + pred_sum + smooth
    dice_per_class = safe_divide(numerator, denominator, zero_division=0.0)
    
    return dice_per_class  # Shape: (batch, num_classes)


# Simple implementations matching your provided functions
def continuous_dice_simple(A_binary, B_probability_map):
    """
    Simple continuous Dice coefficient with c factor.
    
    Args:
        A_binary: Binary ground truth mask
        B_probability_map: Continuous probability predictions
    
    Returns:
        Continuous Dice coefficient (cDC)
    """
    AB = A_binary * B_probability_map
    c = np.sum(AB) / max(np.size(AB[AB > 0]), 1)
    cDC = 2 * (np.sum(AB)) / (c * np.sum(A_binary) + np.sum(B_probability_map))
    return cDC


def dice_coefficient_simple(A_binary, B_binary):
    """
    Simple binary Dice coefficient.
    
    Args:
        A_binary: Binary ground truth mask
        B_binary: Binary predictions
    
    Returns:
        Dice coefficient (DC)
    """
    AB = A_binary * B_binary
    DC = 2 * (np.sum(AB)) / (np.sum(A_binary) + np.sum(B_binary))
    return DC

if __name__ == "__main__":
    # Test with synthetic data
    batch_size, num_classes, height, width = 2, 3, 64, 64
    
    # Create soft predictions (probabilities)
    preds_soft = np.random.rand(batch_size, num_classes, height, width)
    preds_soft = preds_soft / np.sum(preds_soft, axis=1, keepdims=True)  # Normalize to probabilities
    
    # Create ground truth (class indices)
    targets_indices = np.random.randint(0, num_classes, (batch_size, height, width))
    
    # Test continuous dice
    dice_scores = continuous_dice_coefficient(preds_soft, targets_indices, num_classes=num_classes)
    dice_per_class = continuous_dice_per_class(preds_soft, targets_indices, num_classes=num_classes)
    
    print(f"Continuous Dice scores (averaged): {dice_scores}")
    print(f"Continuous Dice scores (per class): {dice_per_class}")
    print(f"Shapes - dice_scores: {dice_scores.shape}, dice_per_class: {dice_per_class.shape}")