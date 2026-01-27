import json
import math
import torch
from torch.utils.data import Dataset, DataLoader
from src.dataset.dataset import ADE20KDataset, make_val_transform
from src.utils.params import Params, integrate_global_parameters
from src.utils.metrics import compute_miou, compute_pixel_accuracy
from src.utils.drawing import color_np_mask
from src.utils.losses import choose_loss
from pathlib import Path
import PIL.Image as Image
import platform
import numpy as np

def validate_model(model : torch.nn.Module, 
                   dataset : Dataset, 
                   params : Params, 
                   output_directory="./output/validation_results/",
                   save_info: bool = True):
    """
    Validate the model on the given dataset using specified parameters.

    Args:
        model: The machine learning model to be validated.
        dataset: The dataset on which to validate the model.
        params: A dictionary of parameters for validation.
        output_directory: Directory to save validation outputs.
        save_info: Whether to save additional information during validation.

    Returns:
        validation_results: Results of the validation process.
    """
    # Implementation of the validation logic goes here
    device = params.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    output_directory = output_directory + model.__class__.__name__ + "/"
    path = Path(output_directory)
    path.mkdir(parents=True, exist_ok=True)
    # Handle multi-GPU with DataParallel
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    else:
        print("Using single GPU or CPU")
        # Optional: Compile model for faster execution (PyTorch 2.0+)
        if device.type == 'cuda' and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
                print("Model compiled with torch.compile()")
            except Exception as e:
                print(f"torch.compile failed: {e}")
    
    model.to(device)
    model.eval()
    num_workers = params.get("num_workers", 4)
    # Handle Windows multiprocessing issues
    if platform.system() == 'Windows' and num_workers > 0:
        print("Warning: On Windows, using num_workers=0 to avoid multiprocessing issues")
        num_workers = 0
    
    pin_memory = device.type == 'cuda'
    batch_size = params.get("batch_size", 8)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=pin_memory)


    losses = params.get("validation_loss_functions", {})
    losses_params = params.get("validation_loss_params", {})
    losses = {key: choose_loss(loss_name, losses_params) for key, loss_name in losses.items()}
    loss_list = {key: 0.0 for key in losses.keys()}
    total_miou = 0.0
    total_pixel_acc = 0.0
    num_batches = 0
    num_classes = params.get("num_classes", 151)
    ignore_index = params.get("ignore_index", 0)  # ignore background by default
    
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs = data['image'].to(device)
            annotation = data['annotation'].to(device)
            
            # Get logits from forward() for loss computation
            if hasattr(model, 'module'):
                logits = model.module.forward(inputs)
            else:
                logits = model.forward(inputs)

            # Skip batches where all pixels are ignored to avoid NaNs
            if ignore_index is not None:
                valid_pixels = (annotation != ignore_index).sum().item()
                if valid_pixels == 0:
                    continue

            # Compute losses (requires logits, not class predictions)
            for key, loss_fn in losses.items():
                loss = loss_fn(logits, annotation) 
                if torch.isnan(loss):
                    continue
                loss_list[key] += loss.item()
            
            # Get predictions from predict() for metrics computation
            if hasattr(model, 'module'):
                outputs = model.module.predict(inputs)
            else:
                outputs = model.predict(inputs)
            # Save output images for the first batch only
            if(save_info and i == 0):
                output_np = outputs.cpu().numpy()
                annotation_np = annotation.cpu().numpy()
                batch_size_current = inputs.size(0)
                for b in range(batch_size_current):
                    color_output = color_np_mask(output_np[b])
                    color_annotation = color_np_mask(annotation_np[b])
                    # color_np_mask already returns RGB in [0, 255]
                    combined_image = np.concatenate((color_annotation, color_output), axis=1)
                    output_image = Image.fromarray(combined_image.astype(np.uint8))
                    output_image.save(f"{output_directory}/validation_result_{i*batch_size + b}.png")
            # Compute metrics
            miou, _ = compute_miou(outputs, annotation, num_classes, ignore_index=ignore_index)
            pixel_acc = compute_pixel_accuracy(outputs, annotation, ignore_index=ignore_index)
            if not math.isnan(miou):
                total_miou += miou
            if not math.isnan(pixel_acc):
                total_pixel_acc += pixel_acc
            num_batches += 1
    
    # Average losses and metrics
    results = {key: loss_list[key] / num_batches for key in loss_list.keys()}
    results['mIoU'] = total_miou / num_batches if num_batches > 0 else 0.0
    results['pixel_accuracy'] = total_pixel_acc / num_batches if num_batches > 0 else 0.0
    if save_info:
        with open(f"{output_directory}/validation_results.json", "w") as f:
            json.dump(results, f, indent=4)
        print(f"Validation results saved to {output_directory}")

    return results