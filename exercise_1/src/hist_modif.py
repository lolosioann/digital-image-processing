from typing import Dict

import numpy as np

from hist_utils import (
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
    """
    if img_array.ndim != 2:
        raise ValueError("Input image must be 2D.")
    if not np.issubdtype(img_array.dtype, np.floating):
        raise TypeError("Input image must be of float type.")
    if not np.all((0.0 <= img_array) & (img_array <= 1.0)):
        raise ValueError("All image values must be in [0, 1].")
    if not isinstance(hist_ref, dict):
        raise TypeError("hist_ref must be a dictionary.")
    if not all(
        isinstance(k, float) and 0.0 <= k <= 1.0 for k in hist_ref.keys()
    ):
        raise ValueError(
            "Keys in hist_ref must be floats in the range [0, 1]."
        )
    if mode not in ["greedy", "non-greedy", "post-disturbance"]:
        raise NotImplementedError(
            "Supported modes: 'greedy', 'non-greedy', 'post-disturbance'."
        )

    # --- POST DISTURBANCE ---
    if mode == "post-disturbance":
        unique_vals = np.unique(img_array)
        if len(unique_vals) < 2:
            raise ValueError(
                "Image must contain at least 2 unique gray levels "
                "for post-disturbance."
            )

        unique_vals.sort()
        delta = unique_vals[1] - unique_vals[0]  # assume uniform quantization

        noise = np.random.uniform(-delta / 2, delta / 2, size=img_array.shape)
        img_array = img_array + noise
        img_array = np.clip(img_array, 0.0, 1.0)
        mode = "greedy"  # apply greedy after noise addition

    # Histogram calculation
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

    print(
        "Number of unique output levels used:",
        len(set(modification_transform.values())),
    )
    return apply_hist_modification_transform(img_array, modification_transform)


def perform_hist_eq(img_array: np.ndarray, mode: str) -> np.ndarray:
    """
    Perform histogram equalization on a grayscale image using actual gray
    levels.

    Args:
        img_array (np.ndarray): A 2D float array with grayscale
        values in [0, 1].
        mode (str): Histogram modification mode ('greedy' or 'non-greedy').

    Returns:
        np.ndarray: Equalized grayscale image.
    """
    unique_vals = np.unique(img_array)
    L = len(unique_vals)

    hist_ref = {float(v): 1.0 / L for v in unique_vals}

    return perform_hist_modification(img_array, hist_ref=hist_ref, mode=mode)


def perform_hist_matching(
    img_array: np.ndarray, img_array_ref: np.ndarray, mode: str
) -> np.ndarray:
    """
    Perform histogram matching between two grayscale images.

    Args:
        img_array (np.ndarray): A 2D float array with grayscale
        values in [0, 1].
        img_array_ref (np.ndarray): A reference grayscale image.
        mode (str): Histogram modification mode ('greedy' or 'non-greedy').

    Returns:
        np.ndarray: Processed image with matched histogram.
    """
    hist_ref = calculate_hist_of_img(img_array_ref, return_normalized=True)
    return perform_hist_modification(img_array, hist_ref=hist_ref, mode=mode)
