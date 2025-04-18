from typing import Dict

import numpy as np

from hist_utils import calculate_hist_of_img, dict_to_hist_array


def perform_hist_modification(
    img_array: np.ndarray, hist_ref: Dict[float, float], mode: str
) -> np.ndarray:
    """
    Modifies the histogram of a grayscale image to match a given
    target histogram.

    Args:
        img_array (np.ndarray): A 2D float array with grayscale values
        in [0, 1].
        hist_ref (Dict[float, float]): Target histogram as
        {gray_level: frequency},
        where gray_level ∈ [0, 1].
        mode (str): One of ['greedy', 'non-greedy', 'post-disturbance'].

    Returns:
        np.ndarray: A new 2D array with modified pixel values to
        match target histogram.
    """
    # -------------------------------------------------------------------------
    # Input Validation
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Histogram Processing
    # -------------------------------------------------------------------------
    hist_ref_arr = dict_to_hist_array(hist_ref)

    if np.sum(hist_ref_arr) == 0:
        raise ValueError(
            "The histogram reference cannot have zero total frequency."
        )

    hist_ref_arr /= np.sum(hist_ref_arr)  # Normalize to sum to 1
    cdf_ref = np.cumsum(hist_ref_arr)

    # -------------------------------------------------------------------------
    # Image Quantization & Histogram Calculation
    # -------------------------------------------------------------------------
    # Map pixel values to [0, 255] integer levels
    img_levels = np.clip(np.round(img_array * 255), 0, 255).astype(int)

    # Get image histogram and its CDF
    hist_img_dict = calculate_hist_of_img(img_array, return_normalized=True)
    hist_img_arr = dict_to_hist_array(hist_img_dict)
    cdf_img = np.cumsum(hist_img_arr)

    # -------------------------------------------------------------------------
    # Histogram Modification Modes
    # -------------------------------------------------------------------------
    if mode == "greedy":
        # For each level in input, find closest level in target CDF
        mapping = np.zeros(256, dtype=np.uint8)
        for g1 in range(256):
            diff = np.abs(cdf_img[g1] - cdf_ref)
            mapping[g1] = np.argmin(diff)

        # Apply the mapping
        modified_img = mapping[img_levels] / 255.0
        return modified_img.astype(np.float32)

    elif mode == "non-greedy":
        raise NotImplementedError("Non-greedy mode is not yet implemented.")

    elif mode == "post-disturbance":
        raise NotImplementedError(
            "Post-disturbance mode is not yet implemented."
        )

    else:
        raise NotImplementedError(f"Mode '{mode}' is not implemented.")
