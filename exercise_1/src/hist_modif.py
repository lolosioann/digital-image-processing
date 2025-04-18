from typing import Dict

import numpy as np
from PIL import Image

from hist_utils import (
    calculate_hist_of_img,
    dict_to_hist_array,
    show_histogram,
)


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

    hist_ref_arr = dict_to_hist_array(hist_ref)
    if np.sum(hist_ref_arr) == 0:
        raise ValueError(
            "The histogram reference cannot have zero total frequency."
        )

    hist_ref_arr /= np.sum(hist_ref_arr)
    cdf_ref = np.cumsum(hist_ref_arr)

    if mode == "post-disturbance":
        d = 1.0 / 64
        noise = np.random.uniform(-d / 2, d / 2, size=img_array.shape)
        print(noise)

        img_array = np.clip(img_array + noise, 0.0, 1.0)
        show_histogram(
            calculate_hist_of_img(img_array, return_normalized=True),
            "Disturbed Image Histogram",
        )
        mode = "greedy"

    img_levels = np.clip(np.round(img_array * 255), 0, 255).astype(int)

    hist_img_dict = calculate_hist_of_img(img_array, return_normalized=True)
    hist_img_arr = dict_to_hist_array(hist_img_dict)
    cdf_img = np.cumsum(hist_img_arr)

    if mode == "greedy":
        # mapping = np.zeros(256, dtype=np.uint8)
        mapping = np.zeros(len(cdf_img), dtype=np.uint8)
        # for g1 in range(256):
        for g1 in range(len(cdf_img)):
            diff = np.abs(cdf_img[g1] - cdf_ref)
            mapping[g1] = np.argmin(diff)
            # mapping.append(np.argmin(diff))

        print(img_levels)
        modified_img = mapping[img_levels] / 255.0
        return modified_img.astype(np.float32)

    elif mode == "non-greedy":
        total_pixels = img_array.size
        ideal_bin_count = total_pixels / 256
        counts = np.round(hist_img_arr * total_pixels).astype(int)

        mapping = np.full(256, -1, dtype=int)
        assigned = np.zeros(256, dtype=int)

        input_levels = np.argsort(cdf_img)
        output_levels = np.argsort(cdf_ref)

        i = 0  # index for input levels
        for j in range(256):  # output level
            current_sum = 0
            while i < 256 and (
                current_sum + counts[input_levels[i]] <= ideal_bin_count
                or abs(ideal_bin_count - current_sum)
                < counts[input_levels[i]] / 2
            ):
                mapping[input_levels[i]] = output_levels[j]
                current_sum += counts[input_levels[i]]
                assigned[output_levels[j]] += counts[input_levels[i]]
                i += 1

        # For any remaining input levels
        for k in range(256):
            if mapping[k] == -1:
                mapping[k] = output_levels[np.argmin(assigned)]
                assigned[mapping[k]] += counts[k]

        modified_img = mapping[img_levels] / 255.0
        return modified_img.astype(np.float32)

    # elif mode == "post-disturbance":
    #     d = 1.0 / 255
    #     noise = np.random.uniform(-d / 2, d / 2, size=img_array.shape)
    #     disturbed_img = np.clip(img_array + noise, 0.0, 1.0)

    #     disturbed_levels = np.clip(np.round(disturbed_img * 255), 0, 255).astype(np.uint8)

    #     hist_img_dict = calculate_hist_of_img(disturbed_img, return_normalized=True)
    #     hist_img_arr = dict_to_hist_array(hist_img_dict)
    #     cdf_img = np.cumsum(hist_img_arr)

    #     mapping = np.zeros(256, dtype=np.uint8)
    #     for g1 in range(256):
    #         diff = np.abs(cdf_img[g1] - cdf_ref)
    #         mapping[g1] = np.argmin(diff)

    #     modified_img = mapping[disturbed_levels] / 255.0
    #     return modified_img.astype(np.float32)

    else:
        raise NotImplementedError(f"Mode '{mode}' is not implemented.")
