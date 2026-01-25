import os
import mlflow
import torch
from torchvision.transforms import v2
from torchvision.tv_tensors import Mask

from ..train import  model_trainer
from ..utils.losses import choose_loss
from ..utils.params import Params, integrate_global_parameters
from ..dataset.dataset import ADE20KDataset, get_random_subset
def verify_model_file(model_path):
    """Verify that the model file exists and is in a valid format."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"The model file at {model_path} does not exist.")
    
    valid_extensions = ['.pt', '.pth', '.h5', '.onnx']
    _, ext = os.path.splitext(model_path)
    ext = ext.lower()
    if ext not in valid_extensions:
        raise ValueError(f"The model file extension {ext} is not supported. Supported extensions are: {valid_extensions}")
    
    print(f"The model file at {model_path} is verified and valid.")
    return True

def _make_memorization_transform(mean, std, size=(256, 256)):
    """No-augmentation transform for memorization sanity checks.
    Uses only a center crop and normalization; wraps masks with tv_tensors.Mask
    to preserve class indices through spatial ops.
    """
    center_crop = v2.CenterCrop(size=size)

    def _transform(image, mask):
        # Ensure mask is treated as label tensor
        mask = Mask(mask)
        image = center_crop(image)
        mask = center_crop(mask)

        image = v2.ToImage()(image)
        image = v2.ToDtype(torch.float32, scale=True)(image)
        image = v2.Normalize(mean=mean, std=std)(image)

        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze().to(dtype=torch.long)  # Remove singleton dims (e.g., channel=1)
        else:
            mask = torch.as_tensor(mask, dtype=torch.long).squeeze()
        return image, mask

    return _transform

def verify_model_memorization(model, dataset, sample_size=10, params : Params | None = None, epochs=100, no_augment: bool = True):
    """Verify if the model can memorize a small subset of the dataset.

    If `no_augment` is True, rebuild a dataset with a no-augmentation transform
    to provide a clean learning signal for memorization tests.
    """
    mlflow.set_experiment("Testing_model_overfitting")

    # IMPORTANT: Enable system metrics monitoring
    mlflow.config.enable_system_metrics_logging()
    mlflow.config.set_system_metrics_sampling_interval(1)
    if params is None:
        params = Params()
    params = integrate_global_parameters(params)
    params.set('num_epochs', epochs)

    # Optionally replace dataset transforms with a no-augmentation pipeline
    if no_augment:
        mean = params.get("mean", (0.485, 0.456, 0.406))
        std = params.get("std", (0.229, 0.224, 0.225))
        mem_transform = _make_memorization_transform(mean, std, size=(256, 256))
        dataset = ADE20KDataset(dataset.image_folder, dataset.annotation_folder, transform=mem_transform)

    subset = get_random_subset(dataset, sample_size)
    trainer = model_trainer(model)
    trainer.train_model(subset, subset, parameters=params)
    train_loss = trainer.validate_1_epoch({'CombinedLoss': choose_loss('CombinedLoss', {})}, device=params.get('device', 'cpu'))

    if train_loss['CombinedLoss'] < 1e-4:
        print("Model successfully memorized the small subset of data.")
        return True
    else:
        print("Model failed to memorize the small subset of data.")
        return False
    