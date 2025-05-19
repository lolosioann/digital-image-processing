import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import rescale

from circ_hough import circ_hough
from log_edge import log_edge
from sobel_edge import sobel_edge


def save_figure(fig, filename: str, folder: str = "fig"):
    """
    Save a matplotlib figure to a specified folder.

    Args:
        fig: Matplotlib figure object.
        filename (str): Name of the file to save.
        folder (str): Folder name to save the figure in.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig_path = os.path.join(folder, filename)
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"Saved figure to {fig_path}")


def load_and_preprocess_image(path: str, downscale: float = 0.5) -> np.ndarray:
    """
    Load an image from disk, convert to grayscale,
    and downscale if requested.

    Args:
        path (str): File path to the image.
        downscale (float): Scaling factor to reduce image
        size (for efficiency).

    Returns:
        np.ndarray: Grayscale image as 2D array.
    """
    img = imread(path)

    print(img.shape)
    # Convert RGBA to RGB if needed
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]  # Discard alpha channel

    # Convert to grayscale if RGB
    if img.ndim == 3:
        img = rgb2gray(img)

    # Downscale image for computational efficiency
    if downscale < 1.0:
        img = rescale(img, downscale, anti_aliasing=True)

    return img


def plot_sobel_thresholds(img: np.ndarray, thresholds: list):
    """
    Visualize Sobel edge maps in a 2x2 grid for different threshold
    values and plot edge pixel counts.

    Args:
        img (np.ndarray): Grayscale input image.
        thresholds (list): List of 4 float thresholds to apply.
    """
    assert (
        len(thresholds) == 4
    ), "Threshold list must contain exactly 4 elements for 2x2 grid."

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()  # Flatten for easy iteration
    num_edges = []

    for i, th in enumerate(thresholds):
        time_start = time()
        edge_img = sobel_edge(img, th)
        time_end = time()
        dt = time_end - time_start
        print(f"Sobel edge detection with threshold {th} took {dt:.4f} s.")
        num_edges.append(np.sum(edge_img))
        axes[i].imshow(edge_img, cmap="gray")
        axes[i].set_title(f"Threshold = {th}")
        axes[i].axis("off")

    fig.suptitle("Sobel Edge Detection with Different Thresholds", fontsize=16)
    fig.subplots_adjust(top=0.85)  # Adjust title position

    plt.tight_layout()
    plt.show()
    save_figure(fig, "sobel_thresholds.png")
    plt.close(fig)

    fig2 = plt.figure()
    plt.plot(thresholds, num_edges, marker="o")
    plt.title("Number of Edge Pixels vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Edge Pixels Count")
    plt.grid(True)
    plt.show()
    save_figure(fig2, "sobel_edge_count.png")
    plt.close(fig2)


def plot_log_sigmas(img: np.ndarray, sigmas: list):
    """
    Visualize LoG edge maps in a 2x2 grid for different
    sigma values and plot edge pixel counts.

    Args:
        img (np.ndarray): Grayscale input image.
        sigmas (list): List of 4 float sigma values to apply.
    """
    assert (
        len(sigmas) == 4
    ), "Sigma list must contain exactly 4 elements for 2x2 grid."

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    num_edges = []

    for i, sigma in enumerate(sigmas):
        time_start = time()
        edge_img = log_edge(img, sigma=sigma)
        time_end = time()
        dt = time_end - time_start
        print(f"LoG edge detection with sigma {sigma} took {dt:.4f} s.")
        num_edges.append(np.sum(edge_img))
        axes[i].imshow(edge_img, cmap="gray")
        axes[i].set_title(f"Sigma = {sigma}")
        axes[i].axis("off")

    fig.suptitle("LoG Edge Detection with Different Sigma Values", fontsize=16)
    fig.subplots_adjust(top=0.85)  # Adjust title position

    plt.tight_layout()
    plt.show()
    save_figure(fig, "log_sigmas.png")
    plt.close(fig)

    fig2 = plt.figure()
    plt.plot(sigmas, num_edges, marker="o")
    plt.title("Number of Edge Pixels vs Sigma")
    plt.xlabel("Sigma")
    plt.ylabel("Edge Pixels Count")
    plt.grid(True)
    plt.show()
    save_figure(fig2, "log_edge_count.png")
    plt.close(fig2)


def compare_sobel_and_log(
    img: np.ndarray, sobel_thres: float, log_sigma: float
):
    """
    Compare Sobel and Laplacian of Gaussian (LoG) edge detection visually.

    Args:
        img (np.ndarray): Grayscale image.
        sobel_thres (float): Threshold for Sobel edge detection.
        log_sigma (float): Standard deviation for LoG kernel.
    """
    sobel_img = sobel_edge(img, sobel_thres)
    log_img = log_edge(img, sigma=log_sigma)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(sobel_img, cmap="gray")
    axes[0].set_title(f"Sobel Edge (thres={sobel_thres})")
    axes[0].axis("off")

    axes[1].imshow(log_img, cmap="gray")
    axes[1].set_title(f"LoG Edge (sigma={log_sigma})")
    axes[1].axis("off")

    plt.suptitle("Comparison of Sobel and LoG Edge Detection", fontsize=16)
    plt.subplots_adjust(top=0.85)  # Adjust title position
    plt.tight_layout()
    plt.show()
    save_figure(fig, "sobel_vs_log.png")
    plt.close()


def display_detected_circles(
    img: np.ndarray,
    edge_img: np.ndarray,
    R_max: float,
    dim: np.ndarray,
    V_mins: list,
):
    """
    Detect circles for multiple vote thresholds and display results in
    a grid layout.

    Args:
        img (np.ndarray): Original grayscale image.
        edge_img (np.ndarray): Binary edge image (e.g. from LoG or Sobel).
        R_max (float): Maximum search radius for circle detection.
        dim (np.ndarray): Quantization of the Hough space [Nx, Ny, Nr].
        V_mins (list): List of vote thresholds to test for circle detection.
    """

    num_vmins = len(V_mins)
    ncols = 2
    nrows = int(np.ceil(num_vmins / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    axes = axes.ravel()

    for i, V_min in enumerate(V_mins):
        time_start = time()
        centers, radii = circ_hough(edge_img, R_max, dim, V_min, R_min=15)
        time_end = time()
        dt = time_end - time_start
        print(f"Hough Circle detection with V_min {V_min} took {dt:.4f} s.")

        ax = axes[i]
        ax.imshow(img, cmap="gray")

        for (a, b), r in zip(centers, radii):
            circle = Circle((a, b), r, color="red", fill=False, linewidth=1)
            ax.add_patch(circle)

        ax.set_title(f"Hough Circles (V_min = {V_min})")
        ax.axis("off")

    # Hide unused axis
    for j in range(len(V_mins), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Detected Circles with Different Vote Thresholds", fontsize=16
    )
    fig.subplots_adjust(top=0.85)  # Adjust title position

    plt.tight_layout()
    plt.show()
    save_figure(fig, "hough_circles_grid.png")
    plt.close(fig)


def main():
    """
    Main driver function to demonstrate edge and circle detection.
    Loads image, applies edge detection, performs Hough transform,
    and visualizes results.
    """
    img_path = "basketball_large.png"
    img = load_and_preprocess_image(img_path, downscale=0.5)

    # Sobel analysis
    thresholds = [0.2, 0.3, 0.5, 0.7]
    print("Plotting Sobel results for different thresholds...")
    plot_sobel_thresholds(img, thresholds)

    # LoG analysis
    log_sigmas = [2, 3, 4, 5]
    print("Plotting LoG results for different thresholds...")
    plot_log_sigmas(img, log_sigmas)

    # Sobel vs LoG comparison
    print("Comparing Sobel and LoG edge detection...")
    compare_sobel_and_log(img, sobel_thres=0.5, log_sigma=4.0)

    # Circle detection with different vote thresholds
    print("Running Hough Circle detection... (this may take a while)")
    edge_img = log_edge(img, sigma=4.0)  # Use LoG for edge detection

    R_max = 1000  # Maximum radius for Hough transform (in pixels)
    dim = np.array([img.shape[1], img.shape[0], 40])  # Hough space bins
    V_mins = [25, 30, 35, 40]  # List of vote thresholds to test
    display_detected_circles(img, edge_img, R_max, dim, V_mins)


if __name__ == "__main__":
    main()
