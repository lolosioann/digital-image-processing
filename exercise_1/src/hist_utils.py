from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def calculate_hist_of_img(
    img_array: np.ndarray, return_normalized: bool = False
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
    img_array: np.ndarray, modification_transform: Dict[float, float]
) -> np.ndarray:
    if img_array.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale array")

    if np.any((img_array < 0) | (img_array > 1)):
        raise ValueError("Pixel values must be between 0 and 1")

    unique_vals = np.unique(img_array)
    for val in unique_vals:
        if val not in modification_transform:
            raise ValueError(
                f"Level {val} does not exist in the modification transform"
            )

    transform_func = np.vectorize(
        lambda x: modification_transform[np.round(x, 4)], otypes=[np.float64]
    )
    modified_img = transform_func(img_array)

    return modified_img


def dict_to_hist_array(hist_dict: Dict[float, float]) -> np.ndarray:
    """
    Converts a histogram stored as a dictionary into a 256-length numpy array.
    Missing keys are treated as 0.
    """
    hist_arr = np.zeros(256, dtype=np.float64)
    for k, v in hist_dict.items():
        hist_arr[int(k * 255)] = v
    return hist_arr


def show_histogram(
    hist_dict: Dict[int, float], title: str = "Histogram"
) -> None:
    # Convert histogram dictionary to array
    hist_arr = dict_to_hist_array(hist_dict)

    # Normalize the histogram array to sum to 1 (probability density)
    hist_arr /= np.sum(hist_arr)

    # Create the plot
    plt.bar(
        np.arange(256) / 255.0,
        hist_arr,
        width=3 / 255.0,
        color="blue",
        alpha=0.7,
    )

    # Set title and axis labels
    plt.title(title)
    plt.xlabel("Pixel Value (Normalized)")
    plt.ylabel("Probability Density")

    # Set the x-axis and y-axis limits to [0, 1]
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Show the plot
    plt.show()
