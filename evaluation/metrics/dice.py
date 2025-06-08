from typing import Dict, Union, Literal
from torch import Tensor
import torch
import torch.nn.functional as F
from torchmetrics.segmentation import DiceScore
import numpy as np

def continuous_dice_coefficient(preds, targets, ignore_index=True, num_classes=2, smooth=1e-6):
    """Computes the continuous Dice coefficient between soft predictions and targets using NumPy.
    
    Args:
        preds: Soft predictions/probabilities. Can be:
               - Raw logits (will be softmaxed)
               - Probabilities (should sum to 1 across class dimension)
               Shape: (batch, num_classes, H, W) or (batch, H, W, num_classes) or (batch, H, W)
        targets: Ground truth masks/labels. Can be:
                - One-hot encoded: (batch, num_classes, H, W)
                - Class indices: (batch, H, W) - will be one-hot encoded
        ignore_index: Whether to ignore background class (index 0)
        num_classes: Number of classes in the dataset
        smooth: Small number to avoid division by 0
        
    Returns:
        Continuous Dice score per batch
    """
    
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
    
    # Handle different input formats for predictions
    if preds.ndim == 4:
        # Check if channel dimension is last and move to second position
        if preds.shape[-1] == num_classes and preds.shape[1] != num_classes:
            preds = np.moveaxis(preds, -1, 1)  # (batch, H, W, C) -> (batch, C, H, W)
    
    # Apply softmax if predictions don't appear to be probabilities or normalized continous values 
    # (assuming they're logits if any value is > 1 or < 0, or don't sum to ~1)
    if len(preds.shape) == 4:  # (batch, C, H, W)
        prob_sums = np.sum(preds, axis=1)  # Sum across class dimension
        if np.any(preds < 0) or np.any(preds > 1) or not np.allclose(prob_sums, 1.0, atol=0.1):
            preds = softmax(preds, axis=1)
    
    # Handle targets - convert to one-hot if needed
    if targets.ndim == 3 and preds.ndim == 4:  # Class indices format (batch, H, W)
        targets = targets.astype(np.int64)
        targets = np.clip(targets, 0, num_classes - 1)
        targets_onehot = np.eye(num_classes)[targets]  # (batch, H, W, C)
        targets = np.moveaxis(targets_onehot, -1, 1)  # (batch, C, H, W)
    elif targets.ndim == 4:  # Already one-hot or continuous
        if targets.shape[-1] == num_classes and targets.shape[1] != num_classes:
            targets = np.moveaxis(targets, -1, 1)  # (batch, H, W, C) -> (batch, C, H, W)
        targets = targets.astype(np.float32)
    
    # Ensure both have same shape
    assert preds.shape == targets.shape, f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}"
    
    # Optionally ignore background class
    if ignore_index and preds.shape[1] > 1:
        preds, targets = ignore_background(preds, targets)
    
    # Compute intersection and union for continuous case
    # Spatial dimensions to reduce over (everything except batch and class)
    reduce_axes = tuple(range(2, preds.ndim))
    
    # Continuous intersection: element-wise product summed over spatial dimensions
    intersection = preds * targets  # Element-wise product (batch, classes, H, W)
    intersection_sum = np.sum(intersection, axis=reduce_axes)  # (batch, classes)
    
    # Compute c factor: mean value of intersection region in probability map
    # c = sum(intersection) / max(count_of_positive_intersection, 1)
    c_values = np.zeros_like(intersection_sum)
    for b in range(preds.shape[0]):  # batch dimension
        for cls in range(preds.shape[1]):  # class dimension
            intersect_region = intersection[b, cls]
            positive_count = np.sum(intersect_region > 0)
            if positive_count > 0:
                c_values[b, cls] = np.sum(intersect_region) / positive_count
            else:
                c_values[b, cls] = 1.0  # Default to 1 if no intersection
    
    # Continuous sums
    pred_sum = np.sum(preds, axis=reduce_axes)  # (batch, classes)
    target_sum = np.sum(targets, axis=reduce_axes)  # (batch, classes)
    
    # Compute continuous Dice coefficient as defined in https://www.biorxiv.org/content/10.1101/306977v1.full.pdf 
    # cDC = 2 * sum(intersection) / (c * sum(binary) + sum(probability))
    numerator = 2.0 * intersection_sum
    denominator = c_values * target_sum + pred_sum + smooth
    
    dice_per_class = safe_divide(numerator, denominator, zero_division=0.0)
    
    # Return mean across classes for each batch item
    dice_score = np.mean(dice_per_class, axis=-1)  # (batch,)
    
    return dice_score

def dice_coefficient_torchmetrics(preds, targets, ignore_index=True, num_classes=2, smooth=1e-6):
    """Computes the Dice coefficient between predictions and targets using NumPy.
    Args:
        targets: ground truth masks/labels
        preds: Binarized predictions
        num_classes: Number of classes in the dataset
        smooth: Small number to avoid division with 0 in denominator
    Returns:
        Dice value per scalar or per batch
    """
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

    # Convert to numpy arrays and ensure correct types
    preds = np.asarray(preds)
    targets = np.asarray(targets).astype(np.int64)
    
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
    
    # Optionally ignore background
    if ignore_index:
        preds, targets = ignore_background(preds, targets)
    
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