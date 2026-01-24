import torch
import torch.nn.functional as F


def compute_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int, 
                 ignore_index: int | None = None) -> tuple[float, torch.Tensor]:
    """
    Compute Mean Intersection over Union (mIoU) for semantic segmentation.
    
    Args:
        pred: Predictions (N, C, H, W) logits or (N, H, W) class indices
        target: Ground truth (N, H, W) class indices
        num_classes: Number of classes
        ignore_index: Class index to ignore (e.g., void class)
    
    Returns:
        Tuple of (mIoU, per_class_iou tensor)
    """
    # If pred has channel dim, convert to class indices
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)  # (N, H, W)
    
    # Flatten
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Create valid mask
    if ignore_index is not None:
        valid_mask = target != ignore_index
        pred = pred[valid_mask]
        target = target[valid_mask]
    
    # Compute per-class IoU
    iou_per_class = torch.zeros(num_classes, device=pred.device)
    class_count = 0
    
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        
        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        
        if union > 0:
            iou_per_class[c] = intersection / union
            class_count += 1
        else:
            iou_per_class[c] = float('nan')  # Class not present
    
    # Mean IoU (only over classes that exist in ground truth)
    # Replace NaN with 0 for missing classes, then average over all classes
    iou_per_class[torch.isnan(iou_per_class)] = 0.0
    miou = iou_per_class.mean().item()
    
    return miou, iou_per_class


def compute_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, 
                           ignore_index: int | None = None) -> float:
    """
    Compute pixel-wise accuracy.
    
    Args:
        pred: Predictions (N, C, H, W) logits or (N, H, W) class indices
        target: Ground truth (N, H, W) class indices
        ignore_index: Class index to ignore
    
    Returns:
        Pixel accuracy as float
    """
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)
    
    if ignore_index is not None:
        valid_mask = target != ignore_index
        correct = ((pred == target) & valid_mask).sum().float()
        total = valid_mask.sum().float()
    else:
        correct = (pred == target).sum().float()
        total = target.numel()
    
    return (correct / total).item() if total > 0 else 0.0
