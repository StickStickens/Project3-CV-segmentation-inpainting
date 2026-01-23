import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
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
    """Create a joint train transform for image/mask using v2 ops."""
    resize_crop = v2.RandomResizedCrop(size=(256, 256), antialias=True)
    hflip = v2.RandomHorizontalFlip(p=0.5)

    def _transform(image, mask):
        # Apply same spatial transforms to both
        seed = torch.randint(0, 2**32, (1,)).item()
        torch.manual_seed(seed)
        image = resize_crop(image)
        torch.manual_seed(seed)
        mask = resize_crop(mask)

        torch.manual_seed(seed)
        image = hflip(image)
        torch.manual_seed(seed)
        mask = hflip(mask)

        # Only normalize the image
        image = v2.ToImage()(image)
        image = v2.ToDtype(torch.float32, scale=True)(image)
        image = v2.Normalize(mean=mean, std=std)(image)

        # Convert mask to integer labels
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return image, mask

    return _transform

def make_val_transform(mean, std):
    """Create a joint val transform for image/mask using v2 ops."""
    center_crop = v2.CenterCrop(size=(256, 256))

    def _transform(image, mask):
        # Apply same spatial transform to both
        image = center_crop(image)
        mask = center_crop(mask)

        # Only normalize the image
        image = v2.ToImage()(image)
        image = v2.ToDtype(torch.float32, scale=True)(image)
        image = v2.Normalize(mean=mean, std=std)(image)

        # Convert mask to integer labels
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return image, mask

    return _transform

def get_random_subset(dataset, sample_size):
    """Get a random subset of the dataset of given sample size."""
    total_size = len(dataset)
    if sample_size > total_size:
        raise ValueError("Sample size cannot be larger than dataset size.")
    indices = torch.randperm(total_size)[:sample_size]
    subset = torch.utils.data.Subset(dataset, indices)
    return subset