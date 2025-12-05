import numpy as np


def image_to_graph(img_array: np.ndarray) -> np.ndarray:
    """
    Converts an image to a fully-connected undirected
    graph represented by an affinity matrix.

    Args:
        img_array (np.ndarray): A float-type array of
        shape (M, N, C) with values in [0, 1].

    Returns:
        np.ndarray: A float-type 2D array of shape (M*N, M*N),
          representing the symmetric affinity matrix.
    """
    M, N, C = img_array.shape
    num_pixels = M * N

    # Flatten the image to shape (M*N, C), where each
    # row is a pixel's intensity vector
    flat_img = img_array.reshape((num_pixels, C))

    # Efficient pairwise Euclidean distance computation using broadcasting
    diff = (
        flat_img[:, np.newaxis, :] - flat_img[np.newaxis, :, :]
    )  # Shape (num_pixels, num_pixels, C)
    dist_squared = np.sum(
        diff**2, axis=2
    )  # Shape (num_pixels, num_pixels), squared Euclidean distance

    # Convert squared distances to affinity values
    affinity_mat = np.exp(-dist_squared)

    return affinity_mat
