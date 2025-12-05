import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from src.utils.paths import data_path_str
import os

from src.part3.image_to_graph import image_to_graph
from src.part3.spectral_clustering import spectral_clustering


def demo_path(p: str) -> str:
    """
    Resolve a demo-local relative path to an absolute repo path.
    Use like: imread(demo_path("images/my_image.png"))
    """
    if not isinstance(p, str):
        return p
    if os.path.isabs(p):
        return p
    pkg = (__package__.split(".")[-1]) if __package__ else "part3"
    return data_path_str("src", pkg, *p.split("/")).__str__()


def visualize_clustering(image, labels, k, title):
    """
    Visualize clustering result by reshaping labels
    to image shape and assigning colors.
    """
    h, w, _ = image.shape
    segmented = np.zeros((h * w, 3))

    cmap = plt.colormaps.get_cmap("tab10")  # Get base colormap
    colors = [cmap(i / k) for i in range(k)]  # Discretize colormap into k colors

    for cluster_id in range(k):
        segmented[labels == cluster_id] = colors[cluster_id][:3]

    segmented = segmented.reshape((h, w, 3))
    plt.imshow(segmented)
    plt.title(title)
    plt.axis("off")


def main():
    data = loadmat(demo_path("dip_hw_3.mat"))
    d2a, d2b = data["d2a"], data["d2b"]

    # Convert input images to affinity graphs
    affinity_a = image_to_graph(d2a)
    affinity_b = image_to_graph(d2b)

    random_state = 1
    images = [d2a, d2b]
    affinities = [affinity_a, affinity_b]
    image_names = ["d2a", "d2b"]

    for img, affinity, name in zip(images, affinities, image_names):
        plt.figure(figsize=(16, 4))

        # Show original image
        plt.subplot(1, 4, 1)
        plt.imshow(img)
        plt.title(f"{name}: Original")
        plt.axis("off")

        for idx, k in enumerate(range(2, 5), start=2):
            print(f"Running spectral clustering on {name} with k={k}...")
            labels = spectral_clustering(affinity, k, random_state=random_state)
            h, w, _ = img.shape
            labels = labels.reshape((h * w,))
            plt.subplot(1, 4, idx)
            visualize_clustering(img, labels, k, title=f"k={k}")

        plt.suptitle(f"Spectral Clustering Results for {name}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
