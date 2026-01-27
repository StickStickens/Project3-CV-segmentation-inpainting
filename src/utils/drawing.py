import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def color_np_mask(mask, ax=None):
    """
    mask: np.ndarray of shape (H, W) with integer labels
    Draws mask with visually highly distinguishable colors per label.
    Class 0 will also have a unique color.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be 2D (H, W)")

    num_labels = int(mask.max()) + 1  # include class 0

    # Generate HSV colors
    hsv_colors = np.zeros((num_labels, 3))
    hsv_colors[:, 0] = np.linspace(0, 1, num_labels, endpoint=False)  # hue
    hsv_colors[:, 1] = 0.7 + 0.2 * (np.arange(num_labels) % 2)        # alternate saturation slightly
    hsv_colors[:, 2] = 0.7 + 0.2 * ((np.arange(num_labels) // 2) % 2) # alternate brightness

    rgb_colors = mcolors.hsv_to_rgb(hsv_colors)

    # Map mask to RGB
    color_mask = rgb_colors[mask]

    return color_mask

def unnormalize_image(image: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    """
    Un-normalizes an image tensor using the provided mean and std.
    
    Args:
        image: Tensor of shape (C, H, W) or (B, C, H, W)
        mean: List of means for each channel
        std: List of standard deviations for each channel

    Returns:
        Un-normalized image tensor of the same shape as input.
    """
    return image * np.array(std).reshape(1,1, -1) + np.array(mean).reshape(1, 1, -1)