import numpy as np
from typing import Dict

def calculate_hist_of_img(
        img_array: np.ndarray,
        return_normalized: bool = False
) -> Dict:
    
    if img_array.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale array")
    if np.any((img_array < 0) | (img_array > 1)):
        raise ValueError("Pixel values must be between 0 and 1")
    
    # flat = np.round(img_array.flatten(), decimals=4)
    unique_vals, counts = np.unique(img_array.flatten(), return_counts=True)

    if return_normalized:
        counts = counts / counts.sum()

    return dict(zip(unique_vals, counts))



def apply_hist_modification_transform(
    img_array: np.ndarray,
    modification_transform: Dict[float, float]
) -> np.ndarray:
    
    if img_array.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale array")
    
    if np.any((img_array < 0) | (img_array > 1)):
        raise ValueError("Pixel values must be between 0 and 1")

    unique_vals = np.unique(img_array)
    for val in unique_vals:
        if val not in modification_transform:
            raise ValueError(f"Level {val} does not exist in the modification transform")

    transform_func = np.vectorize(lambda x: modification_transform[np.round(x, 4)],
                                  otypes=[np.float64])
    modified_img = transform_func(img_array)

    return modified_img
