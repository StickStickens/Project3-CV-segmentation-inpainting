from .validate_model import validate_model
import torch
from torch.utils.data import Dataset
from src.utils.params import Params
from pathlib import Path
import pandas as pd
from src.dataset.dataset import get_random_subset
from src.utils.drawing import color_np_mask, unnormalize_image
from src.models.model_abstract import ModelAbstract
import PIL.Image as Image
from PIL import ImageDraw, ImageFont
from torchvision.transforms import v2


def compare_models(models : list[torch.nn.Module], dataset : Dataset, params : Params, 
                   output_directory="./output/comparison_results/"):
        """
        Compare multiple models on the given dataset using specified parameters.
        Args:
            models: A list of machine learning models to be compared.
            dataset: The dataset on which to compare the models.
            params: A dictionary of parameters for comparison.
            output_directory: Directory to save comparison outputs.
        Returns:
            comparison_results: A list of results for each model.
        """

        comparison_results = []
        for model in models:
            print(f"Validating model: {model.__class__.__name__}")
            results = validate_model(model, dataset, params, output_directory + "comparison/" + model.__class__.__name__ + "/",
                                     save_info=False)
            comparison_results.append({
                "model_name": model.__class__.__name__,
                "results": results
            })

        # Save comparison results to a CSV file
        path = Path(output_directory)
        path.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame([{
            "model_name": res["model_name"],
            **res["results"]
        } for res in comparison_results])
        results_df.to_csv(path / "comparison_results.csv", index=False)
        return comparison_results

def compare_models_visually(models : list[ModelAbstract], dataset : Dataset, params : Params, 
                             output_directory="./output/comparison_visuals/",
                             sample_number : int = 5):
        """
        Compare multiple models visually on the given dataset using specified parameters.
        Args:
            models: A list of machine learning models to be compared.
            dataset: The dataset on which to compare the models.
            params: A dictionary of parameters for comparison.
            output_directory: Directory to save comparison visual outputs.
            sample_number: Number of samples to visualize for each model.
        """
        subset = get_random_subset(dataset, sample_number)
        path = Path(output_directory)
        path.mkdir(parents=True, exist_ok=True, mode = 0o777)
        model_names = [model.__class__.__name__ for model in models]
        image_reducer = v2.Resize(size=(params.get("input_height", 256), params.get("input_width", 256)))
        for i, sample in enumerate(subset):
            output_file = path / f"comparison_{i}.png"
            path.mkdir(parents=True, exist_ok=True, mode = 0o777)
            output_list = []
            input_image = sample['image'].numpy().transpose(1, 2, 0)  # C,H,W to H,W,C
            gt_mask = None
            if 'annotation' in sample:
                gt_mask = sample['annotation'].numpy().astype('uint8')
            for model in models:
                # Get image as tensor for processing
                image_tensor = sample['image'].unsqueeze(0)  # Add batch dim: (C,H,W) -> (1,C,H,W)
                image_tensor = model.fit_to_size(image_tensor)
                
                # Predict and get output
                output = model.predict(image_tensor)  # Returns (H,W) or (1,H,W)
                print(output.shape)
                if output.ndim == 3:
                    output = output.squeeze(0)  # Remove batch dim if present
                
                # Resize to match input image size
                output = torch.nn.functional.interpolate(
                    output.unsqueeze(0).unsqueeze(0).float(), 
                    size=input_image.shape[:2], 
                    mode='nearest'
                ).squeeze(0).squeeze(0).long()
                
                output_np = output.cpu().numpy().astype('uint8')
                output_colored = color_np_mask(output_np)
                output_list.append((model.__class__.__name__, output_colored))
            # Save combined image
            print(f"Combined image size will be calculated for sample {i}")
            # Add space for labels at the top
            label_height = input_image.shape[0] // 10  # 10% of image height for labels
            combined_width = input_image.shape[1] + sum([out.shape[1] for _, out in output_list])
            combined_height = max(input_image.shape[0], max([out.shape[0] for _, out in output_list]))
            if(gt_mask is not None):
                combined_height = max(combined_height, gt_mask.shape[0])
                combined_width += gt_mask.shape[1]
            
            # Create combined image with extra height for labels
            combined_image = Image.new('RGB', (combined_width, combined_height + label_height), color='white')
            draw = ImageDraw.Draw(combined_image)
            
            # Try to load a font, fallback to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", label_height // 2)
            except:
                font = ImageFont.load_default()
            
            # Paste input image and add label
            input_image = (unnormalize_image(input_image, params.get("mean", [0.485, 0.456, 0.406]), 
                                                        params.get("std", [0.229, 0.224, 0.225])) * 255).astype('uint8')
            input_pil = Image.fromarray(input_image)
            combined_image.paste(input_pil, (0, label_height))
            draw.text((input_image.shape[1] // 2, label_height // 2), "Input Image", 
                     fill='black', anchor='mm', font=font)
            
            current_x = input_image.shape[1]
            for model_name, output in output_list:
                # print(output[:5, :5, 0])
                output = (output * 255).astype('uint8')
                output_pil = Image.fromarray(output, mode='RGB')
                combined_image.paste(output_pil, (current_x, label_height))
                # Add model name label above the output
                draw.text((current_x + output.shape[1] // 2, label_height // 2), model_name, 
                         fill='black', anchor='mm', font=font)
                current_x += output.shape[1]
            
            if(gt_mask is not None):
                mask_colored = color_np_mask(gt_mask)
                print(mask_colored[1,1])
                mask_colored = (mask_colored * 255).astype('uint8')
                print(mask_colored.shape)
                gt_pil = Image.fromarray(mask_colored)
                combined_image.paste(gt_pil, (current_x, label_height))
                draw.text((current_x + gt_mask.shape[1] // 2, label_height // 2), "Ground Truth Mask", 
                         fill='black', anchor='mm', font=font)
            
            combined_image.save(output_file)
            

            
