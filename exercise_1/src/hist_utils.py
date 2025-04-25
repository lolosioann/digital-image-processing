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
    # Validate input image
    if img_array is None:
        raise ValueError("Input image cannot be None")
    if img_array.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale array")
    if np.any((img_array < 0) | (img_array > 1)):
        raise ValueError("Pixel values must be between 0 and 1")

    # Compute histogram (unique pixel values and their counts)
    unique_vals, counts = np.unique(img_array.flatten(), return_counts=True)

    # If normalization is requested, convert counts to probability distribution
    if return_normalized:
        counts = counts / counts.sum()

    # Return histogram as a dictionary (pixel value -> frequency)
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
    # Validate input image
    if img_array is None:
        raise ValueError("Input image cannot be None")
    if img_array.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale array")
    if np.any((img_array < 0) | (img_array > 1)):
        raise ValueError("Pixel values must be between 0 and 1")

    # Ensure all unique values are present in the modification transform
    unique_vals = np.unique(img_array)
    for val in unique_vals:
        if val not in modification_transform:
            raise ValueError(
                f"Level {val} does not exist in the modification transform"
            )

    # Apply transformation using vectorized function
    transform_func = np.vectorize(
        lambda x: modification_transform[x], otypes=[np.float64]
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
    # Initialize histogram array with 256 bins
    hist_arr = np.zeros(256, dtype=np.float64)

    # Map each key-value pair from the dictionary to the
    # corresponding bin in the array
    for k, v in hist_dict.items():
        bin_index = int(round(k * 255))  # Scale the float keys to [0, 255]
        hist_arr[bin_index] = v

    return hist_arr


def show_histogram(
    hist_dict: Dict[float, float],
    title: str = "Histogram",
    save_path: str = None,
) -> None:
    """
    Displays or saves a histogram as a bar plot, normalized to [0, 1].

    Args:
        hist_dict (Dict[float, float]): Histogram data with float keys in
            [0, 1].
        title (str): Plot title.
        save_path (str, optional): If provided, saves the plot to this path.
    """
    # Convert histogram dictionary to a 256-bin array
    hist_arr = dict_to_hist_array(hist_dict)

    # Normalize the histogram to make it a probability distribution (sum = 1)
    total = hist_arr.sum()
    if total > 0:
        hist_arr /= total
    else:
        raise ValueError("Cannot plot histogram with zero total frequency")

    # Plot the histogram using a bar plot
    plt.figure()
    plt.bar(
        np.linspace(0, 1, 256),  # x-axis: pixel values between 0 and 1
        hist_arr,  # y-axis: normalized frequency
        width=1 / 255.0,  # Bar width for each bin
        color="blue",  # Bar color
        alpha=0.7,  # Transparency level of bars
    )
    plt.title(title)  # Set plot title
    plt.xlabel("Pixel Value (Normalized)")  # Label for x-axis
    plt.ylabel("Probability Density")  # Label for y-axis
    plt.xlim(0, 1)  # Set x-axis range
    plt.grid(
        True, linestyle="--", alpha=0.3
    )  # Add grid for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping

    # If save_path is provided, save the plot as an image
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()  # Otherwise, display the plot
