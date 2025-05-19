from typing import Tuple

import numpy as np
from numpy.lib.stride_tricks import as_strided


def fir_conv(
    in_img_array: np.ndarray,
    h: np.ndarray,
    in_origin: np.ndarray = None,
    mask_origin: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies a 2D FIR (Finite Impulse Response) filter to a 2D grayscale image
    using vectorized convolution with kernel flipping and origin handling.

    Args:
        in_img_array (np.ndarray): A 2D grayscale image with values normalized
            in the range [0, 1]. Must be of shape (H, W).
        h (np.ndarray): A 2D filter kernel (FIR mask) used for convolution.
        in_origin (np.ndarray, optional): Coordinates of the origin in the
            input image. Defaults to [0, 0] if not provided.
        mask_origin (np.ndarray, optional): Coordinates of the origin in the
            filter mask. Defaults to the center of the mask.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - out_img (np.ndarray): The filtered image of the same shape as
              the input image.
            - out_origin (np.ndarray): The new origin in the output image
              adjusted for the filtering operation.

    Raises:
        ValueError: If input constraints are violated (e.g., non-2D image,
        pixel values out of bounds, or invalid kernel).
    """
    # Input validation
    if in_img_array is None:
        raise ValueError("Input image cannot be None")
    if in_img_array.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale array")
    if np.any((in_img_array < 0) | (in_img_array > 1)):
        raise ValueError("Pixel values must be in the range [0, 1]")
    if h is None or h.ndim != 2:
        raise ValueError("Filter kernel must be a 2D array")

    # Retrieve input and kernel dimensions
    img_h, img_w = in_img_array.shape
    kernel_h, kernel_w = h.shape

    # Compute padding sizes
    pad_h, pad_w = kernel_h // 2, kernel_w // 2

    # Flip the kernel for true convolution
    h_flipped = np.flip(h)

    # Zero-padding around the image
    padded_img = np.pad(
        in_img_array,
        pad_width=((pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
    )

    # Define the shape and strides for the strided window view
    strided_shape = (img_h, img_w, kernel_h, kernel_w)
    strided_strides = (
        padded_img.strides[0],  # stride for image row
        padded_img.strides[1],  # stride for image column
        padded_img.strides[0],  # stride within kernel row
        padded_img.strides[1],  # stride within kernel column
    )

    # Create a view into the padded image using strided windows
    windows = as_strided(
        padded_img, shape=strided_shape, strides=strided_strides
    )

    # Perform vectorized convolution (dot product over each window)
    # Einstein summation notation:
    # "ijkl,kl->ij" means sum over k and l dimensions
    # of the window and kernel, resulting in a 2D output
    # where i and j are the output image dimensions
    # and k and l are the kernel dimensions.
    out_img = np.einsum("ijkl,kl->ij", windows, h_flipped)

    # Assign default origins if not provided
    if in_origin is None:
        in_origin = np.array([0, 0])
    if mask_origin is None:
        mask_origin = np.array([pad_h, pad_w])

    # Compute new origin after convolution
    out_origin = in_origin + mask_origin - np.array([pad_h, pad_w])

    return out_img, out_origin
