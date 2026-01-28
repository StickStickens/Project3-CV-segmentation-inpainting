import platform
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils.params import Params
from src.utils.drawing import color_np_mask
from pathlib import Path
import PIL.Image as Image
import numpy as np

def predict_dataset(model: torch.nn.Module, dataset: Dataset, batch_size: int = 8,
            output_directory: str = "./output/", num_workers: int = 4) -> None:
    """ Function to perform prediction on a given dataset using the provided model.
        Args:
            model: The trained PyTorch model for prediction.
            dataset: The dataset on which to perform predictions.
            batch_size: The batch size for DataLoader.
            output_directory: Directory to save prediction outputs. (it will create a subfolder with model class name)
            num_workers: Number of DataLoader workers.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    
    # Handle Windows multiprocessing issues
    if platform.system() == 'Windows' and num_workers > 0:
        print("Warning: On Windows, using num_workers=0 to avoid multiprocessing issues")
        num_workers = 0
    
    pin_memory = device.type == 'cuda'
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=pin_memory)


    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device, non_blocking=True)
        annotation = batch['annotation'].to(device, non_blocking=True)
        with torch.no_grad():
            # Handle both wrapped (DataParallel) and unwrapped models
            if hasattr(model, 'module'):
                outputs = model.module.predict(images)
            else:
                outputs = model.predict(images)
        # Process outputs as needed (e.g., convert to CPU, save, etc.)
        outputs = outputs.cpu()
        # Save each output mask
        for i in range(outputs.size(0)):
            output_mask = outputs[i].numpy().astype('uint8')  # Class indices 0-150
            save_path = path / f"output_{batch_idx * batch_size + i}.png"
            # Save the output mask - values are class indices (grayscale)
            original_mask = batch['annotation'][i].cpu().numpy().astype('uint8')
            merged_output = np.concatenate((original_mask, output_mask), axis=1)
            output_mask = color_np_mask(merged_output)
            print(output_mask.shape)
            if output_mask.dtype != np.uint8:
                output_mask = (output_mask*255).astype(np.uint8)
            output_image = Image.fromarray(output_mask, mode='RGB')
            output_image.save(save_path)
    print(f"Predictions saved to {output_directory}")

def predict_image(model: torch.nn.Module, image: np.ndarray,
                  output_path: str = "./output_image.png", save_image: bool = True) -> None:
    """ Function to perform prediction on a single image using the provided model.
        Args:
            model: The trained PyTorch model for prediction.
            image: The input image as a NumPy array (H, W, C).
            output_path: Path to save the prediction output.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Preprocess image
    input_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)  # (1, C, H, W)
    
    with torch.no_grad():
        # Handle both wrapped (DataParallel) and unwrapped models
        if hasattr(model, 'module'):
            output = model.module.predict(input_tensor)
        else:
            output = model.predict(input_tensor)
    
    output = output.cpu().squeeze(0).numpy().astype('uint8')  # Class indices 0-150
    # Save the output mask
    if save_image:
        output_image = color_np_mask(output)
        output_image = Image.fromarray(output_image, mode='RGB')
        output_image.save(output_path)
        print(f"Prediction saved to {output_path}")
    return output