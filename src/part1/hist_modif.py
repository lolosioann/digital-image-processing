from typing import Dict

import numpy as np

from .hist_utils import (
    apply_hist_modification_transform,
    calculate_hist_of_img,
)


def perform_hist_modification(
    img_array: np.ndarray, hist_ref: Dict[float, float], mode: str
) -> np.ndarray:
    """
    Modifies the histogram of a grayscale image to match a given
    target histogram using the specified mode.
    Supported modes: 'greedy', 'non-greedy', 'post-disturbance'.

    Args:
        img_array (np.ndarray): The grayscale input image (2D array
            with values in [0, 1]).
        hist_ref (Dict[float, float]): The target histogram as a
            dictionary with grayscale values as keys and their frequency
            as values.
        mode (str): The mode of histogram modification ('greedy',
            'non-greedy', 'post-disturbance').

    Returns:
        np.ndarray: The modified image array with the target histogram.
    """

    # Input validations
    if img_array.ndim != 2:
        raise ValueError("Input image must be 2D.")
    if not np.issubdtype(img_array.dtype, np.floating):
        raise TypeError("Input image must be of float type.")
    if not np.all((0.0 <= img_array) & (img_array <= 1.0)):
        raise ValueError("All image values must be in [0, 1].")
    if not isinstance(hist_ref, dict):
        raise TypeError("hist_ref must be a dictionary.")
    if not all(isinstance(k, float) and 0.0 <= k <= 1.0 for k in hist_ref.keys()):
        raise ValueError("Keys in hist_ref must be floats in the range [0, 1].")
    if len(hist_ref) == 0:
        raise ValueError("Target histogram must contain at least one gray level.")
    if mode not in ["greedy", "non-greedy", "post-disturbance"]:
        raise NotImplementedError(
            "Supported modes: 'greedy', 'non-greedy', 'post-disturbance'."
        )

    # post disturbance mode
    if mode == "post-disturbance":
        unique_vals = np.unique(img_array)

        # we need at least 2 levels to calculate delta
        if len(unique_vals) < 2:
            raise ValueError(
                "Image must contain at least 2 unique gray levels "
                "for post-disturbance."
            )

        unique_vals.sort()
        delta = unique_vals[1] - unique_vals[0]  # Assume uniform quantization
        noise = np.random.uniform(-delta / 2, delta / 2, size=img_array.shape)
        img_array = img_array + noise
        img_array = np.clip(img_array, 0.0, 1.0)

        mode = "greedy"  # Apply greedy after noise addition

    # Histogram calculation for input image
    input_hist = calculate_hist_of_img(img_array, return_normalized=False)
    N = sum(input_hist.values())

    sorted_input_levels = sorted(input_hist.keys())
    sorted_output_levels = sorted(hist_ref.keys())
    desired_counts = {g: int(round(freq * N)) for g, freq in hist_ref.items()}
    modification_transform = {}

    if mode == "greedy":
        input_samples = []
        for val in sorted_input_levels:
            input_samples.extend([val] * input_hist[val])
        input_samples.sort()

        output_samples = []
        for g in sorted_output_levels:
            output_samples.extend([g] * desired_counts[g])

        len_diff = len(input_samples) - len(output_samples)
        if len_diff > 0:
            output_samples.extend([sorted_output_levels[-1]] * len_diff)
        elif len_diff < 0:
            output_samples = output_samples[: len(input_samples)]

        for i_val, o_val in zip(input_samples, output_samples):
            if i_val not in modification_transform:
                modification_transform[i_val] = o_val

    # --- NON-GREEDY MODE ---
    elif mode == "non-greedy":
        input_pointer = 0
        input_levels = sorted_input_levels
        assigned = set()
        num_levels = len(input_levels)

        for g in sorted_output_levels:
            target_count = desired_counts[g]
            current_count = 0

            while input_pointer < num_levels:
                f = input_levels[input_pointer]
                count_f = input_hist[f]
                deficiency = target_count - current_count

                if deficiency < count_f / 2:
                    break

                modification_transform[f] = g
                current_count += count_f
                assigned.add(f)
                input_pointer += 1

        for f in sorted_input_levels:
            if f not in modification_transform:
                nearest_g = min(sorted_output_levels, key=lambda g: abs(g - f))
                modification_transform[f] = nearest_g

    return apply_hist_modification_transform(img_array, modification_transform)


def perform_hist_eq(img_array: np.ndarray, mode: str) -> np.ndarray:
    """
    Perform histogram equalization on a grayscale image using actual gray
    levels.

    Args:
        img_array (np.ndarray): A 2D float array with grayscale values in
            [0, 1].
        mode (str): Histogram modification mode ('greedy',
            'non-greedy' or 'post-disturbance').

    Returns:
        np.ndarray: Equalized grayscale image.
    """
    hist_ref = {float(v / 256): 1.0 / 256 for v in range(256)}
    return perform_hist_modification(img_array, hist_ref=hist_ref, mode=mode)


def perform_hist_matching(
    img_array: np.ndarray, img_array_ref: np.ndarray, mode: str
) -> np.ndarray:
    """
    Perform histogram matching between two grayscale images.

    Args:
        img_array (np.ndarray): A 2D float array with grayscale values in
            [0, 1].
        img_array_ref (np.ndarray): A reference grayscale image.
        mode (str): Histogram modification mode ('greedy',
            'non-greedy' or 'post-disturbance').

    Returns:
        np.ndarray: Processed image with matched histogram.
    """
    hist_ref = calculate_hist_of_img(img_array_ref, return_normalized=True)
    return perform_hist_modification(img_array, hist_ref=hist_ref, mode=mode)
