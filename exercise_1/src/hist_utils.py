from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def calculate_hist_of_img(
    img_array: np.ndarray, return_normalized: bool = False
) -> Dict[float, float]:
    """
    Computes the histogram of a grayscale image with float values in [0, 1].

    Args:
        img_array (np.ndarray): 2D grayscale image array.
        return_normalized (bool): If True, returns a probability distribution.

    Returns:
        Dict[float, float]: Histogram as a mapping from pixel value
        to frequency.
    """
    if img_array.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale array")
    if np.any((img_array < 0) | (img_array > 1)):
        raise ValueError("Pixel values must be between 0 and 1")

    unique_vals, counts = np.unique(img_array.flatten(), return_counts=True)

    if return_normalized:
        counts = counts / counts.sum()

    return dict(zip(unique_vals, counts))


def apply_hist_modification_transform(
    img_array: np.ndarray, modification_transform: Dict[float, float]
) -> np.ndarray:
    """
    Applies a transformation to an image using a precomputed value mapping.

    Args:
        img_array (np.ndarray): 2D grayscale image array.
        modification_transform (Dict[float, float]): Mapping of
            pixel values to new values.

    Returns:
        np.ndarray: Transformed image array.
    """
    if img_array.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale array")
    if np.any((img_array < 0) | (img_array > 1)):
        raise ValueError("Pixel values must be between 0 and 1")

    # Ensure all unique values are present in the transform
    unique_vals = np.unique(img_array)
    for val in unique_vals:
        if np.round(val, 4) not in modification_transform:
            raise ValueError(
                f"Level {val} does not exist in the modification transform"
            )

    # Apply transformation using vectorized function
    transform_func = np.vectorize(
        lambda x: modification_transform[np.round(x, 4)], otypes=[np.float64]
    )
    return transform_func(img_array)


def dict_to_hist_array(hist_dict: Dict[float, float]) -> np.ndarray:
    """
    Converts a histogram dictionary into a 256-bin numpy array.
    Assumes keys are in [0.0, 1.0].

    Args:
        hist_dict (Dict[float, float]): Histogram as float keys in [0, 1].

    Returns:
        np.ndarray: Histogram array of length 256.
    """
    hist_arr = np.zeros(256, dtype=np.float64)
    for k, v in hist_dict.items():
        bin_index = int(round(k * 255))
        hist_arr[bin_index] = v
    return hist_arr


def show_histogram(
    hist_dict: Dict[float, float], title: str = "Histogram"
) -> None:
    """
    Displays a histogram as a bar plot, normalized to [0, 1] on both axes.

    Args:
        hist_dict (Dict[float, float]): Histogram data with float
            keys in [0, 1].
        title (str): Plot title.
    """
    hist_arr = dict_to_hist_array(hist_dict)

    # Normalize to probability density
    total = hist_arr.sum()
    if total > 0:
        hist_arr /= total
    else:
        raise ValueError("Cannot plot histogram with zero total frequency")

    # Plot histogram: x-axis in [0, 1], y-axis in [0, 1]
    plt.bar(
        np.linspace(0, 1, 256),
        hist_arr,
        width=1 / 255.0,
        color="blue",
        alpha=0.7,
    )

    # Labels and plot formatting
    plt.title(title)
    plt.xlabel("Pixel Value (Normalized)")
    plt.ylabel("Probability Density")
    plt.xlim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()
