import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks."""
    
    def __init__(self, smooth=1.0, ignore_index: int | None = None, **kwargs):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        """
        Args:
            pred: predictions of shape (N, C, H, W) - logits or probabilities
            target: ground truth of shape (N, H, W) - class indices
        """
        num_classes = pred.shape[1]
        pred = F.softmax(pred, dim=1)
        
        # Create mask for valid pixels
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).unsqueeze(1).float()  # (N, 1, H, W)
        else:
            valid_mask = torch.ones_like(target).unsqueeze(1).float()
        
        # Clamp target for one-hot (ignore_index might be out of range)
        target_clamped = target.clone()
        if self.ignore_index is not None:
            target_clamped[target == self.ignore_index] = 0
        
        # Convert target to one-hot encoding
        target_one_hot = F.one_hot(target_clamped, num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # Apply valid mask
        pred = pred * valid_mask
        target_one_hot = target_one_hot * valid_mask
        
        # Flatten
        pred = pred.contiguous().view(pred.shape[0], num_classes, -1)
        target_one_hot = target_one_hot.contiguous().view(target_one_hot.shape[0], num_classes, -1)
        
        # Compute dice coefficient
        intersection = (pred * target_one_hot).sum(dim=2)
        union = pred.sum(dim=2) + target_one_hot.sum(dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()
        
        return dice_loss


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in segmentation."""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', **kwargs):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Args:
            pred: predictions of shape (N, C, H, W) - logits
            target: ground truth of shape (N, H, W) - class indices
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class IoULoss(nn.Module):
    """IoU (Jaccard) Loss for segmentation tasks."""
    
    def __init__(self, smooth=1.0, **kwargs):
        super(IoULoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: predictions of shape (N, C, H, W) - logits or probabilities
            target: ground truth of shape (N, H, W) - class indices
        """
        num_classes = pred.shape[1]
        pred = F.softmax(pred, dim=1)
        
        # Convert target to one-hot encoding
        target_one_hot = F.one_hot(target, num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # Flatten
        pred = pred.contiguous().view(pred.shape[0], num_classes, -1)
        target_one_hot = target_one_hot.contiguous().view(target_one_hot.shape[0], num_classes, -1)
        
        # Compute IoU
        intersection = (pred * target_one_hot).sum(dim=2)
        union = pred.sum(dim=2) + target_one_hot.sum(dim=2) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        iou_loss = 1.0 - iou.mean()
        
        return iou_loss


class TverskyLoss(nn.Module):
    """Tversky Loss - generalization of Dice Loss with adjustable FP/FN trade-off."""
    
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0, **kwargs):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # weight for false positives
        self.beta = beta    # weight for false negatives
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: predictions of shape (N, C, H, W) - logits or probabilities
            target: ground truth of shape (N, H, W) - class indices
        """
        num_classes = pred.shape[1]
        pred = F.softmax(pred, dim=1)
        
        # Convert target to one-hot encoding
        target_one_hot = F.one_hot(target, num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # Flatten
        pred = pred.contiguous().view(pred.shape[0], num_classes, -1)
        target_one_hot = target_one_hot.contiguous().view(target_one_hot.shape[0], num_classes, -1)
        
        # Compute components
        true_pos = (pred * target_one_hot).sum(dim=2)
        false_pos = (pred * (1 - target_one_hot)).sum(dim=2)
        false_neg = ((1 - pred) * target_one_hot).sum(dim=2)
        
        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth)
        tversky_loss = 1.0 - tversky.mean()
        
        return tversky_loss
    
class CombinedCELDiceLoss(nn.Module):
    """Combined CrossEntropy + Dice loss for multiclass segmentation.
    Useful to balance per-pixel classification with overlap optimization.

    Args:
        ce_weight: weight for CrossEntropy component (default 1.0)
        dice_weight: weight for Dice component (default 0.5)
        label_smoothing: optional CE label smoothing (default 0.0)
        ignore_index: optional index to ignore in CE and Dice (e.g., void label)
    """

    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 0.5,
                 label_smoothing: float = 0.0, ignore_index: int | None = None, **kwargs):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index if ignore_index is not None else -100
        self._dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # CrossEntropy expects logits; supports optional smoothing/ignore_index
        ce = F.cross_entropy(pred, target, label_smoothing=self.label_smoothing,
                             ignore_index=self.ignore_index)
        dice = self._dice(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice
    
def choose_loss(loss_name: str, params: dict = None) -> torch.nn.Module:
    """ Function to choose the loss function based on the given name.
        Args:
            loss_name: Name of the loss function as a string.
            params: Dictionary of parameters to pass to the loss function.

        Returns:
            A PyTorch loss function.
    """
    if params is None:
        params = {}
    
    if loss_name == 'CrossEntropyLoss':
        return torch.nn.CrossEntropyLoss()
    elif loss_name == 'DiceLoss':
        return DiceLoss(**params)
    elif loss_name == 'FocalLoss':
        return FocalLoss(**params)
    elif loss_name == 'IoULoss':
        return IoULoss(**params)
    elif loss_name == 'TverskyLoss':
        return TverskyLoss(**params)
    elif loss_name == 'CombinedLoss':
        return CombinedCELDiceLoss(**params)
    elif loss_name == 'MSELoss':
        return torch.nn.MSELoss(**params)
    elif loss_name == 'L1Loss':
        return torch.nn.L1Loss(**params)
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")
