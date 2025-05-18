from typing import Tuple

import numpy as np


def fir_conv(
    in_img_array: np.ndarray,
    h: np.ndarray,
    in_origin: np.ndarray = None,
    mask_origin: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies a 2D FIR filter to a 2D grayscale image.

    Args:
        in_img_array (np.ndarray): 2D grayscale image with
        values in [0, 1].
        h (np.ndarray): 2D FIR filter kernel.
        in_origin (np.ndarray, optional): Origin of input image.
        mask_origin (np.ndarray, optional): Origin of the kernel.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Filtered image and new origin.
    """
    if in_img_array is None:
        raise ValueError("Input image cannot be None")
    if in_img_array.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale array")
    if np.any((in_img_array < 0) | (in_img_array > 1)):
        raise ValueError("Pixel values must be in the range [0, 1]")
    if h is None or h.ndim != 2:
        raise ValueError("Filter kernel must be a 2D array")

    # Input dimensions
    img_h, img_w = in_img_array.shape
    kernel_h, kernel_w = h.shape
    pad_h, pad_w = kernel_h // 2, kernel_w // 2

    # Flip kernel (convolution)
    h_flipped = np.flip(h)

    # Pad image
    padded_img = np.pad(
        in_img_array, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant"
    )

    strided_shape = (img_h, img_w, kernel_h, kernel_w)
    strided_strides = (
        padded_img.strides[0],  # row stride
        padded_img.strides[1],  # col stride
        padded_img.strides[0],  # row stride again for kernel window
        padded_img.strides[1],  # col stride again for kernel window
    )
    from numpy.lib.stride_tricks import as_strided

    windows = as_strided(
        padded_img, shape=strided_shape, strides=strided_strides
    )

    # Vectorized convolution
    out_img = np.einsum("ijkl,kl->ij", windows, h_flipped)

    # Handle origin defaults
    if in_origin is None:
        in_origin = np.array([0, 0])
    if mask_origin is None:
        mask_origin = np.array([pad_h, pad_w])

    # Output origin computation
    out_origin = in_origin + mask_origin - np.array([pad_h, pad_w])

    return out_img, out_origin
