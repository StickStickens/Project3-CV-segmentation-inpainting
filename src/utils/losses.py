import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks."""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
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
        
        # Compute dice coefficient
        intersection = (pred * target_one_hot).sum(dim=2)
        union = pred.sum(dim=2) + target_one_hot.sum(dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()
        
        return dice_loss


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in segmentation."""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
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
    
    def __init__(self, smooth=1.0):
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
    
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
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
    
def choose_loss( loss_name: str):
    """ Function to choose the loss function based on the given name.
        Args:
            loss_name: Name of the loss function as a string.

        Returns:
            A PyTorch loss function.
    """
    if loss_name == 'CrossEntropyLoss':
        return torch.nn.CrossEntropyLoss()
    elif loss_name == 'DiceLoss':
        return DiceLoss()
    elif loss_name == 'FocalLoss':
        return FocalLoss()
    elif loss_name == 'IoULoss':
        return IoULoss()
    elif loss_name == 'TverskyLoss':
        return TverskyLoss()
    elif loss_name == 'MSELoss':
        return torch.nn.MSELoss()
    elif loss_name == 'L1Loss':
        return torch.nn.L1Loss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")
