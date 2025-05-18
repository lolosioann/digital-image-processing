import numpy as np
from scipy.signal import convolve2d


def sobel_edge(in_img_array: np.ndarray, thres: float) -> np.ndarray:
    """
    Detect edges in a grayscale image using the Sobel operator.

    Args:
        in_img_array (np.ndarray): 2D grayscale image with values in [0, 1].
        thres (float): Threshold for gradient magnitude; must be a
            positive value.

    Returns:
        np.ndarray: 2D binary output image with values in {0, 1} indicating
            edge locations.
    """
    if in_img_array is None or in_img_array.ndim != 2:
        raise ValueError("Input must be a 2D grayscale image.")
    if not np.all((0 <= in_img_array) & (in_img_array <= 1)):
        raise ValueError("Input image values must be in the range [0, 1].")
    if thres <= 0:
        raise ValueError("Threshold must be a positive number.")

    # Sobel kernels for horizontal (x) and vertical (y) gradients
    Gx = np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]], dtype=float)

    Gy = np.array([[+1, +2, +1], [0, 0, 0], [-1, -2, -1]], dtype=float)

    # Compute gradient components via convolution
    grad_x = convolve2d(
        in_img_array, Gx, mode="same", boundary="fill", fillvalue=0
    )
    grad_y = convolve2d(
        in_img_array, Gy, mode="same", boundary="fill", fillvalue=0
    )

    # Compute gradient magnitude
    grad_magnitude = np.hypot(
        grad_x, grad_y
    )  # more stable than sqrt(x^2 + y^2)

    # Apply threshold to obtain binary edge map
    out_img_array = (grad_magnitude >= thres).astype(int)

    return out_img_array
