from typing import Tuple

import numpy as np
from scipy.ndimage import maximum_filter


def merge_similar_circles(
    centers: np.ndarray,
    radii: np.ndarray,
    delta_c: float = 10.0,
    delta_r: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge circles with nearby centers and similar radii to reduce redundancy.

    Args:
        centers (np.ndarray): (N, 2) array of circle centers.
        radii (np.ndarray): (N,) array of corresponding radii.
        delta_c (float): Maximum allowed Euclidean distance between centers.
        delta_r (float): Maximum allowed difference between radii.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of merged centers and radii.
    """
    if len(centers) == 0:
        return centers, radii

    merged_centers = []
    merged_radii = []
    used = np.zeros(len(centers), dtype=bool)

    for i in range(len(centers)):
        if used[i]:
            continue

        group = [i]  # start cluster with circle i
        for j in range(i + 1, len(centers)):
            if used[j]:
                continue
            dist = np.linalg.norm(centers[i] - centers[j])
            r_diff = np.abs(radii[i] - radii[j])
            if dist < delta_c and r_diff < delta_r:
                group.append(j)
                used[j] = True
        used[i] = True

        group_centers = centers[group]
        group_radii = radii[group]
        merged_centers.append(np.mean(group_centers, axis=0))
        merged_radii.append(np.mean(group_radii))

    return np.array(merged_centers), np.array(merged_radii)


def circ_hough(
    in_img_array: np.ndarray,
    R_max: float,
    dim: np.ndarray = np.array([100, 100, 20]),
    V_min: int = 10,
    R_min: float = 0.0,
    n_theta: int = 72,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects circles in a binary edge image using the Circular Hough Transform.

    Args:
        in_img_array (np.ndarray): 2D binary image (values 0 or 1)
        containing edge map.
        R_max (float): Maximum radius of circles to detect.
        dim (np.ndarray): Size of the accumulator space [Nx, Ny, Nr].
        V_min (int): Minimum number of votes to validate a detection.
        R_min (float): Minimum radius of circles to retain in final results.
        n_theta (int): Number of discrete angles (θ) for perimeter sampling.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - centers (K, 2): Coordinates of circle centers.
            - radii (K,): Radii of detected circles.
    """
    # Input validation
    if in_img_array.ndim != 2:
        raise ValueError("Input image must be 2D.")
    if not np.issubdtype(in_img_array.dtype, np.integer):
        raise ValueError("Input image must contain integer values.")
    if np.any((in_img_array != 0) & (in_img_array != 1)):
        raise ValueError("Input image must be binary (0 or 1).")

    img_h, img_w = in_img_array.shape
    Nx, Ny, Nr = dim
    accumulator = np.zeros((Nx, Ny, Nr), dtype=np.uint16)

    # Quantization scales for center and radius
    x_scale = img_w / Nx
    y_scale = img_h / Ny
    r_scale = R_max / Nr

    # Find edge coordinates (foreground pixels)
    edge_points = np.argwhere(in_img_array == 1)
    if edge_points.shape[0] == 0:
        return np.empty((0, 2)), np.empty((0,))

    # Precompute circle perimeter sampling angles for performance
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)

    x = edge_points[:, 1].reshape(-1, 1)  # column (horizontal)
    y = edge_points[:, 0].reshape(-1, 1)  # row (vertical)

    # Voting in Hough space
    for r_idx in range(Nr):
        r = (r_idx + 0.5) * r_scale  # radius value
        dx = r * cos_theta
        dy = r * sin_theta

        a = x - dx  # center x = x - r cos(θ)
        b = y - dy  # center y = y - r sin(θ)

        a_idx = (a / x_scale).astype(np.int32)
        b_idx = (b / y_scale).astype(np.int32)

        a_flat = a_idx.ravel()
        b_flat = b_idx.ravel()
        valid = (0 <= a_flat) & (a_flat < Nx) & (0 <= b_flat) & (b_flat < Ny)

        ai = a_flat[valid]
        bi = b_flat[valid]
        ri = np.full_like(ai, r_idx)

        np.add.at(accumulator, (ai, bi, ri), 1)

    # Local maxima in accumulator space
    max_filt = maximum_filter(accumulator, size=(3, 3, 3))
    local_maxima = (accumulator == max_filt) & (accumulator >= V_min)
    votes = np.argwhere(local_maxima)

    if votes.shape[0] == 0:
        return np.empty((0, 2)), np.empty((0,))

    # Map discrete accumulator indices to continuous space
    centers = np.zeros((votes.shape[0], 2), dtype=float)
    radii = np.zeros(votes.shape[0], dtype=float)

    centers[:, 0] = (votes[:, 0] + 0.5) * x_scale  # x = (i + 0.5)*scale
    centers[:, 1] = (votes[:, 1] + 0.5) * y_scale  # y = (j + 0.5)*scale
    radii[:] = (votes[:, 2] + 0.5) * r_scale  # r = (k + 0.5)*scale

    # Merge overlapping/similar circles
    centers, radii = merge_similar_circles(centers, radii, delta_c=10.0, delta_r=5.0)

    # Filter out small-radius detections (probably noise)
    valid_indices = radii >= R_min
    centers = centers[valid_indices]
    radii = radii[valid_indices]

    return centers, radii
