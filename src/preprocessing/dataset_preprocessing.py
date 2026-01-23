from ..dataset.dataset import ADE20KDataset
from src.utils.params import Params
import numpy as np

def get_and_save_mean_std(params: Params, dataset_list: list[ADE20KDataset]):
    means = []
    stds = []
    num_images = 0
    
    for dataset in dataset_list:
        for index in range(len(dataset)):
            # Dataset returns dict with 'image' key (tensor or PIL)
            image = dataset[index]['image']
            if not isinstance(image, np.ndarray):
                image = np.array(image)  # Convert to numpy if needed
            
            # image shape: (C, H, W) or (H, W, C) - check your dataset
            # Assuming (C, H, W):
            image = image.astype(np.float32) / 255.0 if image.max() > 1 else image.astype(np.float32)
            
            # Compute per-channel mean/std
            image_mean = image.reshape(image.shape[0], -1).mean(axis=1)  # (C,)
            image_std = image.reshape(image.shape[0], -1).std(axis=1)    # (C,)
            
            means.append(image_mean)
            stds.append(image_std)
            num_images += 1
    
    # Average across all images
    mean = np.mean(means, axis=0)  # (C,)
    std = np.mean(stds, axis=0)    # (C,)
    
    mean = mean.tolist()
    std = std.tolist()
    print(f"Computed mean: {mean}, std: {std}")
    params.set("mean", mean)
    params.set("std", std)
    params.save("global_params.json")

def preprocess_ade20k(params: Params):
    # Load parameters
    train_image_folder = params.get("train_image_folder")
    train_annotation_folder = params.get("train_annotation_folder")
    validation_image_folder = params.get("validation_image_folder")
    validation_annotation_folder = params.get("validation_annotation_folder")

    # Create dataset without transformations to compute mean and std
    base_dataset = ADE20KDataset(train_image_folder, train_annotation_folder, transform=None)
    val_dataset = ADE20KDataset(validation_image_folder, validation_annotation_folder, transform=None)
    dataset_list = [base_dataset, val_dataset]
    get_and_save_mean_std(params, dataset_list)