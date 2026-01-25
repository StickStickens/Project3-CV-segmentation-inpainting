import torch

@torch.no_grad()
def compute_miou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> tuple[float, torch.Tensor]:
    """
    Vectorized Mean Intersection over Union (mIoU).

    Args:
        pred: (N, C, H, W) logits or (N, H, W) class indices
        target: (N, H, W) ground truth class indices
        num_classes: number of classes
        ignore_index: class to ignore (e.g. 255)

    Returns:
        (mIoU, per_class_iou)
    """
    # First do argmax if needed (while still float), then convert to int
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)
    
    pred = pred.to(torch.int64)
    target = target.to(torch.int64)

    pred = pred.view(-1)
    target = target.view(-1)

    if ignore_index is not None:
        mask = target != ignore_index
        pred = pred[mask]
        target = target[mask]

    # Confusion matrix
    # Rows = GT, Cols = Pred
    k = num_classes
    conf_mat = torch.bincount(
        k * target + pred,
        minlength=k * k
    ).reshape(k, k).float()

    # Intersection = diagonal
    intersection = torch.diag(conf_mat)

    # Union = row sum + col sum - intersection
    gt_sum = conf_mat.sum(dim=1)
    pred_sum = conf_mat.sum(dim=0)
    union = gt_sum + pred_sum - intersection

    # IoU per class
    iou_per_class = intersection / union
    iou_per_class[union == 0] = torch.nan  # class not present

    # Mean IoU (only valid classes)
    miou = torch.nanmean(iou_per_class).item()

    return miou, iou_per_class



@torch.no_grad()
def compute_pixel_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int | None = None,
) -> float:
    """
    Vectorized pixel accuracy.
    """
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)

    pred = pred.view(-1)
    target = target.view(-1)

    if ignore_index is not None:
        mask = target != ignore_index
        pred = pred[mask]
        target = target[mask]

    if target.numel() == 0:
        return 0.0

    return (pred == target).float().mean().item()

