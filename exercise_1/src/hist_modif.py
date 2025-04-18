from typing import Dict

import numpy as np

from hist_utils import calculate_hist_of_img, dict_to_hist_array


def perform_hist_modification(
    img_array: np.ndarray, hist_ref: Dict[float, float], mode: str
) -> np.ndarray:
    """
    Modifies the histogram of a grayscale image to match a given target
        histogram.

    Args:
        img_array (np.ndarray): 2D grayscale float image with values
            in [0, 1].
        hist_ref (Dict): Desired histogram as {gray_level: frequency}, keys
            are floats in [0, 1].
        mode (str): One of ['greedy', 'non-greedy', 'post-disturbance'].

    Returns:
        np.ndarray: The modified image.
    """
    if img_array.ndim != 2:
        raise ValueError("Input image must be 2D.")
    if not np.issubdtype(img_array.dtype, np.floating):
        raise TypeError("Input image must be of float type.")
    if not all(0.0 <= v <= 1.0 for v in np.nditer(img_array)):
        raise ValueError("All image values must be in [0, 1].")
    if not isinstance(hist_ref, dict):
        raise TypeError("hist_ref must be a dictionary.")

    # Validate hist_ref keys (they should be floats in the range [0, 1])
    if not all(
        isinstance(k, float) and 0.0 <= k <= 1.0 for k in hist_ref.keys()
    ):
        raise ValueError(
            "Keys in hist_ref must be floats in the range [0, 1]."
        )

    # Convert hist_ref dictionary to array
    hist_ref_arr = dict_to_hist_array(hist_ref)

    # Ensure the histogram reference is non-zero before normalizing
    hist_sum = np.sum(hist_ref_arr)
    if hist_sum == 0:
        raise ValueError(
            "The histogram reference cannot have zero total frequency."
        )

    hist_ref_arr /= hist_sum  # Normalize to form a distribution
    cdf_ref = np.cumsum(hist_ref_arr)

    # Quantize input image into levels 0–255 for indexing
    img_levels = np.clip(np.round(img_array * 255), 0, 255).astype(int)

    # Compute histogram of input image
    hist_img_dict = calculate_hist_of_img(img_array, return_normalized=True)
    hist_img_arr = dict_to_hist_array(hist_img_dict)
    cdf_img = np.cumsum(hist_img_arr)

    if mode == "greedy":
        # Compute greedy mapping
        mapping = np.zeros(256, dtype=np.float32)
        for g1 in range(256):
            diff = np.abs(cdf_img[g1] - cdf_ref)
            mapping[g1] = np.argmin(diff)

        # Apply mapping and rescale to [0, 1]
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
