import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import rescale

from circ_hough import circ_hough
from log_edge import log_edge
from sobel_edge import sobel_edge


def load_and_preprocess_image(path: str, downscale: float = 0.5) -> np.ndarray:
    img = imread(path)

    # Handle RGBA -> RGB
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]  # Drop alpha channel

    if img.ndim == 3:
        img = rgb2gray(img)

    if downscale < 1.0:
        img = rescale(img, downscale, anti_aliasing=True)

    return img


def plot_sobel_thresholds(img: np.ndarray, thresholds: list):
    fig, axes = plt.subplots(1, len(thresholds), figsize=(15, 5))
    num_edges = []

    for i, th in enumerate(thresholds):
        edge_img = sobel_edge(img, th)
        num_edges.append(np.sum(edge_img))
        axes[i].imshow(edge_img, cmap="gray")
        axes[i].set_title(f"Threshold = {th}")
        axes[i].axis("off")

    plt.figure()
    plt.plot(thresholds, num_edges, marker="o")
    plt.title("Number of Edge Pixels vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Edge Pixels Count")
    plt.grid(True)
    plt.show()


def compare_sobel_and_log(
    img: np.ndarray, sobel_thres: float, log_sigma: float
):
    sobel_img = sobel_edge(img, sobel_thres)
    log_img = log_edge(img, sigma=log_sigma)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(sobel_img, cmap="gray")
    axes[0].set_title(f"Sobel Edge (thres={sobel_thres})")
    axes[0].axis("off")

    axes[1].imshow(log_img, cmap="gray")
    axes[1].set_title(f"LoG Edge (sigma={log_sigma})")
    axes[1].axis("off")

    plt.show()


def display_detected_circles(
    img: np.ndarray,
    edge_img: np.ndarray,
    R_max: float,
    dim: np.ndarray,
    V_mins: list,
):
    from matplotlib.patches import Circle

    for V_min in V_mins:
        centers, radii = circ_hough(edge_img, R_max, dim, V_min, R_min=15)
        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray")
        for (a, b), r in zip(centers, radii):
            circle = Circle((a, b), r, color="red", fill=False, linewidth=1)
            ax.add_patch(circle)
        ax.set_title(f"Hough Circles (V_min = {V_min})")
        ax.axis("off")
        plt.show()


def main():
    img_path = "basketball_large.png"
    # img_path = "detect_circles_8circles.jpg"
    img = load_and_preprocess_image(img_path, downscale=0.5)

    # --- Sobel analysis ---
    thresholds = [0.1, 0.2, 0.3, 0.5, 0.7]
    print("Plotting Sobel results for different thresholds...")
    plot_sobel_thresholds(img, thresholds)

    # --- LoG comparison ---
    print("Comparing Sobel and LoG edge detection...")
    compare_sobel_and_log(img, sobel_thres=0.5, log_sigma=4.0)

    # --- Circle detection with different vote thresholds ---
    print("Running Hough Circle detection...")
    # edge_img = sobel_edge(img, 0.5)  # Use lower threshold or switch to LoG
    edge_img = log_edge(img, sigma=4.0)  # Use LoG for edge detection
    R_max = 1000  # Increase based on expected size
    dim = np.array([img.shape[1], img.shape[0], 40])  # Adjust bins
    V_mins = [35]
    display_detected_circles(img, edge_img, R_max, dim, V_mins)


if __name__ == "__main__":
    main()
