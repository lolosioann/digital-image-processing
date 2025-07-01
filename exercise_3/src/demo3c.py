import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from image_to_graph import image_to_graph
from n_cuts import n_cuts_recursive


def visualize_clustering(image, labels, k, title):
    """
    Visualize clustering result by reshaping labels
      to image shape and assigning colors.
    """
    h, w, _ = image.shape
    segmented = np.zeros((h * w, 3))

    cmap = plt.colormaps.get_cmap("tab10")  # Get base colormap
    colors = [
        cmap(i / k) for i in range(k)
    ]  # Discretize colormap into k colors

    for cluster_id in range(k):
        segmented[labels == cluster_id] = colors[cluster_id][:3]

    segmented = segmented.reshape((h, w, 3))
    plt.imshow(segmented)
    plt.title(title)
    plt.axis("off")


def main():
    data = loadmat("dip_hw_3.mat")
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
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"{name}: Original")
        plt.axis("off")

        print(f"Running recursive n_cuts on {name} until the end...")
        labels = n_cuts_recursive(
            affinity, T1=5, T2=0.2, random_state=random_state
        )
        h, w, _ = img.shape
        labels = labels.reshape((h * w,))
        plt.subplot(1, 2, 2)
        visualize_clustering(img, labels, 2, title="Recursive n_cuts")

        plt.suptitle(f"N_Cuts Results for {name}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
