import torch
from torch.utils.data import Dataset, DataLoader
from src.utils.params import Params
from pathlib import Path
import PIL.Image as Image

def predict(model: torch.nn.Module, dataset: Dataset, device: torch.device, batch_size: int = 8,
            output_directory : str = "./predictions/") -> torch.Tensor:
    """ Function to perform prediction on a given dataset using the provided model.
        Args:
            model: The trained PyTorch model for prediction.
            dataset: The dataset on which to perform predictions.
            device: The device (CPU or GPU) to run the model on.
            batch_size: The batch size for DataLoader.
            output_directory: Directory to save prediction outputs.

        Returns:
            A tensor containing all predictions.
    """

    path = Path(output_directory)
    path.mkdir(parents=True, exist_ok=True)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    else:
        print("Using single GPU or CPU")
        model = torch.compile(model)
    model.to(device)
    pin_memory = True if device.type == 'cuda' else False
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device, non_blocking=True)
        with torch.no_grad():
            outputs = model.predict(images)
        # Process outputs as needed (e.g., convert to CPU, save, etc.)
        outputs = outputs.cpu()
        # Here you can add code to handle the outputs, e.g., save them or return
        for i in range(outputs.size(0)):
            output_mask = outputs[i]
            save_dir = path / f"output_{batch_idx * batch_size + i}.png"
            # Save the output mask (assuming it's a single-channel image)
            output_image = Image.fromarray((output_mask.numpy() * 255).astype('uint8'))
            output_image.save(save_dir)
    print(f"Predictions saved to {output_directory}")
