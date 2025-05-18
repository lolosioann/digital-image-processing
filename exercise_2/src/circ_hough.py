from typing import Tuple

import numpy as np
from scipy.ndimage import maximum_filter
from tqdm import tqdm


def merge_similar_circles(
    centers: np.ndarray,
    radii: np.ndarray,
    delta_c: float = 10.0,
    delta_r: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge circles that are spatially close and have similar radii.

    Args:
        centers (np.ndarray): Array of shape (N, 2) with circle centers.
        radii (np.ndarray): Array of shape (N,) with circle radii.
        delta_c (float): Maximum distance between centers to consider merging.
        delta_r (float): Maximum difference in radii to consider merging.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Deduplicated centers and radii.
    """
    if len(centers) == 0:
        return centers, radii

    merged_centers = []
    merged_radii = []
    used = np.zeros(len(centers), dtype=bool)

    for i in range(len(centers)):
        if used[i]:
            continue
        # Initialize cluster with current circle
        group = [i]
        for j in range(i + 1, len(centers)):
            if used[j]:
                continue
            dist = np.linalg.norm(centers[i] - centers[j])
            r_diff = np.abs(radii[i] - radii[j])
            if dist < delta_c and r_diff < delta_r:
                group.append(j)
                used[j] = True
        used[i] = True
        # Average center and radius of the group
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
    R_min: float = 0.0,  # New parameter: Minimum allowed radius
    n_theta: int = 72,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect circles in a binary edge image using the Hough Transform.

    Args:
        in_img_array (np.ndarray): Input binary edge image (values 0 or 1).
        R_max (float): Maximum radius to search.
        dim (np.ndarray): Quantization of the Hough space: [Nx, Ny, Nr].
        V_min (int): Minimum votes to accept a circle.
        R_min (float): Minimum radius to accept a circle.
        n_theta (int): Number of angular discretizations.
        verbose (bool): Print diagnostic information.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - centers (K × 2): Coordinates of detected centers (a, b)
            - radii (K,): Radii of detected circles
    """
    if in_img_array.ndim != 2:
        raise ValueError("Input image must be 2D.")
    if not np.issubdtype(in_img_array.dtype, np.integer):
        raise ValueError("Input image must contain binary (0 or 1) values.")
    if np.any((in_img_array != 0) & (in_img_array != 1)):
        raise ValueError("Input image must be binary (0 or 1).")

    img_h, img_w = in_img_array.shape
    Nx, Ny, Nr = dim
    accumulator = np.zeros((Nx, Ny, Nr), dtype=np.uint16)

    x_scale = img_w / Nx
    y_scale = img_h / Ny
    r_scale = R_max / Nr

    # Get edge points
    edge_points = np.argwhere(in_img_array == 1)
    num_edges = edge_points.shape[0]
    if num_edges == 0:
        return np.empty((0, 2)), np.empty((0,))

    # Angular discretization
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)

    x = edge_points[:, 1].reshape(-1, 1)
    y = edge_points[:, 0].reshape(-1, 1)

    for r_idx in tqdm(range(Nr), desc="Voting radii"):
        r = (r_idx + 0.5) * r_scale
        dx = r * cos_theta
        dy = r * sin_theta

        a = x - dx
        b = y - dy

        a_idx = (a / x_scale).astype(np.int32)
        b_idx = (b / y_scale).astype(np.int32)

        a_flat = a_idx.ravel()
        b_flat = b_idx.ravel()
        valid = (0 <= a_flat) & (a_flat < Nx) & (0 <= b_flat) & (b_flat < Ny)
        ai = a_flat[valid]
        bi = b_flat[valid]
        ri = np.full_like(ai, r_idx)

        np.add.at(accumulator, (ai, bi, ri), 1)

    # Local maxima in 3D space (Nx x Ny x Nr)
    max_filt = maximum_filter(accumulator, size=(3, 3, 3))
    local_maxima = (accumulator == max_filt) & (accumulator >= V_min)
    votes = np.argwhere(local_maxima)

    if votes.shape[0] == 0:
        return np.empty((0, 2)), np.empty((0,))

    centers = np.zeros((votes.shape[0], 2), dtype=float)
    radii = np.zeros(votes.shape[0], dtype=float)

    centers[:, 0] = (votes[:, 0] + 0.5) * x_scale
    centers[:, 1] = (votes[:, 1] + 0.5) * y_scale
    radii[:] = (votes[:, 2] + 0.5) * r_scale

    # Merge overlapping/similar circles
    centers, radii = merge_similar_circles(
        centers, radii, delta_c=10.0, delta_r=5.0
    )
    # Filter based on R_min
    valid_indices = radii >= R_min
    centers = centers[valid_indices]
    radii = radii[valid_indices]

    return centers, radii
