"""HuggingFace segmentation wrappers (ConvNeXt semantic, DETR panoptic).

These classes follow the ModelAbstract interface: forward -> logits, predict -> masks.
Pass a HuggingFace model id; defaults provided for ADE20K checkpoints when available.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
from .model_abstract import ModelAbstract
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from src.utils.params import Params, integrate_global_parameters


class Convnext(ModelAbstract):
    """ConvNeXt-based semantic segmentation via HuggingFace.

    Pass a valid semantic-segmentation checkpoint. Example ADE20K checkpoints:
    - openmmlab/upernet-convnext-base (default here, ADE20K)
    - openmmlab/upernet-convnext-small (ADE20K)
    (No official facebook/convnext ADE20K checkpoints exist.)
    """

    def __init__(
        self,
        model_id: str = "openmmlab/upernet-convnext-base",
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_id).to(self.device)
        params = integrate_global_parameters(Params())
        mean = params.get("mean", [0.485, 0.456, 0.406])
        std = params.get("std", [0.229, 0.224, 0.225])
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.current_logits = None #store to avoid recalculation if we use both forward and predict

    def _unnormalize_if_needed(self, images: torch.Tensor) -> torch.Tensor:
        """Un-normalize images if they are in normalized range (values outside [0, 1])."""
        # Check if images are normalized (values outside [0, 1])
        if images.min() < -0.01 or images.max() > 1.01:
            # Un-normalize using ImageNet mean and std
            mean = self.mean.to(images.device)
            std = self.std.to(images.device)
            images = images * std + mean
            # Clip to [0, 1] range
            images = torch.clamp(images, 0, 1)
        return images

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        target_size = tuple(images.shape[-2:])
        images = self._unnormalize_if_needed(images)
        # Processor accepts list of tensors (C,H,W) in [0,1] or [0,255]
        # Images are already in [0,1]; skip additional rescale in the processor
        inputs = self.processor(list(images), return_tensors="pt", do_rescale=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        logits = outputs.logits  # (B, C, H/stride, W/stride)
        # Upsample logits to original spatial size for per-pixel loss/metrics
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
        # If the checkpoint has 150 ADE20K classes (no explicit background), add a dummy background
        if logits.shape[1] == 150:
            bg = torch.zeros(logits.size(0), 1, *logits.shape[-2:], device=logits.device, dtype=logits.dtype)
            logits = torch.cat([bg, logits], dim=1)  # (B, 151, H, W) with background at channel 0
        self.current_logits = logits
        return logits  # (B, C, H, W)

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        single = images.ndim == 3
        if(self.current_logits is not None):
            logits = self.current_logits
        else:
            logits = self.forward(images)  # (B, C, H, W) after upsample
        self.current_logits = None
        with torch.no_grad():
            preds = logits.argmax(dim=1)  # (B, H, W)
        return preds.squeeze(0) if single else preds

    def fit_to_size(self, x: torch.Tensor, size: Tuple[int, int] | None = None) -> torch.Tensor:
        if size is None:
            return x
        need_unsq = x.ndim == 3
        if need_unsq:
            x = x.unsqueeze(0)
        out = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return out.squeeze(0) if need_unsq else out
