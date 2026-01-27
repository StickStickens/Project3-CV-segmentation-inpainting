from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from .model_abstract import ModelAbstract
import torch
import numpy as np
from src.utils.params import Params, integrate_global_parameters
# load Mask2Former fine-tuned on ADE20k semantic segmentation
import torch.nn.functional as F
class mask2former(ModelAbstract):
    def __init__(self, num_classes: int = 150):
        """
        Initialize Mask2Former model for ADE20k semantic segmentation.
        
        Args:
            num_classes: Number of object classes (default 150). Note that the model
                        outputs 150 classes (ADE20k classes 1-150), and we add background
                        class 0 in forward() to make it compatible with 151-class format.
        """
        super().__init__()
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
        self.model.to(self.device)
        # ImageNet normalization constants
        params = integrate_global_parameters(Params())
        self.mean = params.get("mean", [0.485, 0.456, 0.406])
        self.std = params.get("std", [0.229, 0.224, 0.225])
        self.mean = torch.tensor(self.mean).view(1, 3, 1, 1)
        self.std = torch.tensor(self.std).view(1, 3, 1, 1)
        self.current_logits = None

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
        """
        Forward pass returning logits for 151 classes (0=background, 1-150=objects).
        
        IMPORTANT: Mask2Former trained on ADE20k outputs 150 classes indexed 0-149,
        which correspond to ADE20k classes 1-150 (wall, building, sky, ..., flag).
        There is NO background class in the model output.
        
        We need to:
        1. Get predictions (indices 0-149 from model)
        2. Shift them by +1 to get ADE20k indices (1-150)
        3. Add a background channel at index 0
        """
        # Handle single image
        if images.ndim == 3:
            images = images.unsqueeze(0)
        
        batch_size = images.shape[0]
        target_size = tuple(images.shape[-2:])  # (H, W)
        
        # Un-normalize if necessary
        images = self._unnormalize_if_needed(images)
        
        # Convert to numpy uint8 for processor
        images_np = images.permute(0, 2, 3, 1).cpu().numpy()
        images_np = (images_np * 255).astype(np.uint8)
        
        # Process images
        inputs = self.processor(images=images_np, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use the processor's post-processing to get semantic segmentation
        # This returns a list of (H, W) tensors with class indices 0-149
        # where 0=wall, 1=building, 2=sky, ..., 149=flag (ADE20k classes 1-150)
        semantic_maps = self.processor.post_process_semantic_segmentation(
            outputs, 
            target_sizes=[target_size] * batch_size
        )
        
        # Convert semantic maps to logits format (B, 151, H, W)
        logits_list = []
        for semantic_map in semantic_maps:
            # semantic_map is (H, W) with values 0-149 (model's class indices)
            # Shift by +1 to get ADE20k indices: 1-150
            ade20k_indices = semantic_map.long() + 1  # Now 1-150
            
            # Create one-hot encoding for 151 classes (0-150)
            # This will put 1s at positions 1-150, leaving position 0 (background) as 0
            one_hot = F.one_hot(ade20k_indices, num_classes=151)  # (H, W, 151)
            one_hot = one_hot.permute(2, 0, 1).float()  # (151, H, W)
            
            # Convert to logits: high confidence where predicted, low elsewhere
            # Use large values to make argmax work correctly
            logits = one_hot * 100.0 + (1 - one_hot) * (-100.0)  # (151, H, W)
            
            logits_list.append(logits)
        
        # Stack into batch
        pixel_logits = torch.stack(logits_list, dim=0).to(self.device)  # (B, 151, H, W)
        
        self.current_logits = pixel_logits
        return pixel_logits
    
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        # Handle single image
        single = images.ndim == 3
        print(images.shape)
        if(self.current_logits is not None):
            logits = self.current_logits
        else:
            logits = self.forward(images)  # (B, C, H, W) after upsample
        self.current_logits = None
        print(logits.shape)
        with torch.no_grad():
            preds = logits.argmax(dim=1)  # (B, H, W)
        return preds.squeeze(0) if single else preds
    
    def fit_to_size(self, x : torch.tensor): #transformer architecture is size agnostic
        return x