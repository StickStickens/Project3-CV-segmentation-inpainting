import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.tv_tensors import Mask
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

class ADE20KDataset(Dataset):
    def __init__(self, image_folder, annotation_folder, transform=None):
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.transform = transform  # callable that takes (image, mask)
        self.image_filenames = sorted(os.listdir(image_folder))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        annotation_filename = image_filename.replace(".jpg", ".png")
        image_path = os.path.join(self.image_folder, image_filename)
        annotation_path = os.path.join(self.annotation_folder, annotation_filename)
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(annotation_path).convert("L")

        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            image = v2.ToImage()(image)
            image = v2.ToDtype(torch.float32, scale=True)(image)
            mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return {'image': image, 'annotation': mask}

def make_train_transform(mean, std):
    """Create a joint train transform for image/mask using v2 ops.
    Ensures masks use NEAREST interpolation to preserve class indices.
    """
    # Spatial transforms applied to both image and mask
    # Use explicit interpolation for the image; masks will be tv_tensors.Mask -> NEAREST
    # Assume images are pre-resized to ~320 px on the shorter side; keep some scale jitter
    spatial = v2.Compose([
        v2.RandomResizedCrop(
            size=(256, 256),
            scale=(0.7, 1.0),  # crop between 70%-100% of pre-resized image
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        v2.RandomHorizontalFlip(p=0.5),
    ])

    # Color transforms only for image
    jitter = v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
    blur = v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2)

    def _transform(image, mask):
        # Wrap mask to signal label semantics to v2 transforms
        mask = Mask(mask)

        # Apply spatial transforms to both simultaneously
        image, mask = spatial(image, mask)

        # Only process the image
        image = jitter(image)
        image = blur(image)
        image = v2.ToImage()(image)
        image = v2.ToDtype(torch.float32, scale=True)(image)
        image = v2.Normalize(mean=mean, std=std)(image)

        # Convert mask to integer labels (Mask comes back as tensor), remove singleton dims
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze().to(dtype=torch.long)
        else:
            mask = torch.as_tensor(np.array(mask), dtype=torch.long).squeeze()
        return image, mask

    return _transform

def make_val_transform(mean, std):
    """Create a joint val transform for image/mask using v2 ops.
    Masks are wrapped as tv_tensors.Mask to ensure safe spatial ops.
    """
    center_crop = v2.CenterCrop(size=(256, 256))

    def _transform(image, mask):
        # Wrap mask to preserve discrete labels
        mask = Mask(mask)

        # Apply same spatial transform to both
        image = center_crop(image)
        mask = center_crop(mask)

        # Only normalize the image
        image = v2.ToImage()(image)
        image = v2.ToDtype(torch.float32, scale=True)(image)
        image = v2.Normalize(mean=mean, std=std)(image)

        # Convert mask to integer labels, remove singleton dims
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze().to(dtype=torch.long)
        else:
            mask = torch.as_tensor(np.array(mask), dtype=torch.long).squeeze()
        return image, mask

    return _transform

def get_random_subset(dataset : Dataset, sample_size: int):
    """Get a random subset of the dataset of given sample size."""
    total_size = len(dataset)
    if sample_size > total_size:
        raise ValueError("Sample size cannot be larger than dataset size.")
    indices = torch.randperm(total_size)[:sample_size]
    subset = torch.utils.data.Subset(dataset, indices)
    return subset