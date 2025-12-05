from math import ceil

import numpy as np

from .fir_conv import fir_conv


def generate_log_kernel(sigma: float) -> np.ndarray:
    """
    Generates a 2D discrete Laplacian of Gaussian (LoG) kernel.

    Args:
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        np.ndarray: 2D LoG kernel.
    """
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")

    k = ceil(3 * sigma)  # kernel extends to +-3*sigma
    # create a grid of coordinates centered at the origin
    # with size (2k+1) x (2k+1)
    x1, x2 = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1))

    norm = (x1**2 + x2**2) / (2 * sigma**2)
    print(norm)
    log_kernel = (-1 / (np.pi * sigma**4)) * (1 - norm) * np.exp(-norm)

    return log_kernel


def log_edge(in_img_array: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Detects edges in a grayscale image using Laplacian of Gaussian (LoG)
    filtering followed by zero-crossing detection.

    Args:
        in_img_array (np.ndarray): Input 2D grayscale image with
        values in [0, 1].
        sigma (float): Standard deviation for the Gaussian smoothing.

    Returns:
        np.ndarray: 2D binary image with values in {0, 1} indicating
        edge locations.
    """
    if in_img_array.ndim != 2:
        raise ValueError("Input image must be a 2D array.")
    if np.any((in_img_array < 0) | (in_img_array > 1)):
        raise ValueError("Image pixel values must be in [0, 1].")
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")

    # Generate LoG kernel
    log_kernel = generate_log_kernel(sigma)

    # Convolve image with LoG kernel
    filtered, _ = fir_conv(in_img_array, log_kernel)

    # Initialize output for zero-crossings
    zero_cross = np.zeros_like(filtered, dtype=int)

    # Detect sign changes with all 8-connected neighbors
    for shift in [
        (-1, 0),
        (1, 0),  # vertical
        (0, -1),
        (0, 1),  # horizontal
        (-1, -1),
        (-1, 1),  # diagonals
        (1, -1),
        (1, 1),
    ]:
        shifted = np.roll(filtered, shift, axis=(0, 1))
        zero_cross |= (filtered * shifted < 0).astype(int)

    # Suppress artificial borders caused by roll
    zero_cross[[0, -1], :] = 0
    zero_cross[:, [0, -1]] = 0

    return zero_cross
