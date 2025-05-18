from math import ceil

import numpy as np
from scipy.ndimage import convolve


def generate_log_kernel(sigma: float) -> np.ndarray:
    """
    Generate a discrete Laplacian of Gaussian (LoG) kernel.

    Args:
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        np.ndarray: 2D LoG kernel.
    """
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")

    k = ceil(3 * sigma)
    x1, x2 = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1))

    norm = (x1**2 + x2**2) / (2 * sigma**2)
    log_kernel = (-1 / (np.pi * sigma**4)) * (1 - norm) * np.exp(-norm)
    return log_kernel


def log_edge(in_img_array: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Edge detection using Laplacian of Gaussian (LoG) and zero-crossings.

    Args:
        in_img_array (np.ndarray): Input 2D grayscale image in [0, 1].
        sigma (float): Standard deviation for Gaussian smoothing.

    Returns:
        np.ndarray: Binary edge map (0 or 1).
    """
    if in_img_array.ndim != 2:
        raise ValueError("Input image must be a 2D array.")
    if np.any((in_img_array < 0) | (in_img_array > 1)):
        raise ValueError("Image pixel values must be in [0, 1].")
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")

    # Convolve image with LoG filter
    log_kernel = generate_log_kernel(sigma)
    filtered = convolve(in_img_array, log_kernel, mode="reflect")

    # Vectorized zero-crossing detection
    zero_cross = np.zeros_like(filtered, dtype=int)

    # Check sign differences with neighbors (horizontal, vertical, diagonals)
    for shift in [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]:
        shifted = np.roll(filtered, shift, axis=(0, 1))
        zero_cross |= (filtered * shifted < 0).astype(int)

    # Suppress borders (due to roll artifacts)
    zero_cross[[0, -1], :] = 0
    zero_cross[:, [0, -1]] = 0

    return zero_cross
