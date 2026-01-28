import cv2
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

def enlarge_mask(mask: np.ndarray, pixels: int = 5) -> np.ndarray:
    """
    Enlarge positive areas (True or 1) in a binary mask.
    
    Args:
        mask: 2D boolean or 0/1 numpy array
        pixels: number of pixels to grow/dilate

    Returns:
        np.ndarray of same shape, dtype=bool
    """
    # ensure binary uint8
    mask_uint8 = mask.astype(np.uint8)  # 0 or 1
    mask_uint8 *= 255  # 0 or 255
    
    # create kernel for dilation
    kernel = np.ones((2*pixels+1, 2*pixels+1), np.uint8)
    
    # dilate
    dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
    
    # convert back to boolean
    return dilated.astype(bool)