import numpy as np

from fir_conv import fir_conv


def sobel_edge(in_img_array: np.ndarray, thres: float) -> np.ndarray:
    """
    Detects edges in a 2D grayscale image using the Sobel operator.

    Args:
        in_img_array (np.ndarray): 2D grayscale image with values in [0, 1].
        thres (float): Positive threshold for the gradient magnitude.

    Returns:
        np.ndarray: 2D binary image with values in {0, 1}
        indicating edge locations.
    """
    if in_img_array is None or in_img_array.ndim != 2:
        raise ValueError("Input must be a 2D grayscale image.")
    if not np.all((0 <= in_img_array) & (in_img_array <= 1)):
        raise ValueError("Input image values must be in the range [0, 1].")
    if thres <= 0:
        raise ValueError("Threshold must be a positive number.")

    # Sobel kernels for horizontal and vertical edges
    Gx = np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]], dtype=float)

    Gy = np.array([[+1, +2, +1], [0, 0, 0], [-1, -2, -1]], dtype=float)

    # Convolve input image with Sobel kernels
    grad_x, _ = fir_conv(in_img_array, Gx)
    grad_y, _ = fir_conv(in_img_array, Gy)

    # Compute gradient magnitude
    grad_magnitude = np.hypot(grad_x, grad_y)  # sqrt(x^2 + y^2)

    # Threshold gradient magnitude to get binary edge map
    out_img_array = (grad_magnitude >= thres).astype(int)

    return out_img_array
