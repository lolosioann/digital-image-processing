import matplotlib.pyplot as plt
import numpy as np
import pytest

from part2.circ_hough import circ_hough


def create_test_circle_image(
    shape=(100, 100), center=(50, 50), radius=20
) -> np.ndarray:
    """Δημιουργεί μια δυαδική εικόνα με έναν κύκλο ακτίνας `radius`."""
    y, x = np.indices(shape)
    mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2
    return mask.astype(int)


def test_no_circle():
    img = np.zeros((100, 100), dtype=int)
    dim = np.array([100, 100, 40])
    R_max = 40
    V_min = 30
    centers, radii = circ_hough(img, R_max, dim, V_min)
    assert len(centers) == 0


def test_plot_detected_circles():
    img = create_test_circle_image(center=(40, 60), radius=15)
    dim = np.array([100, 100, 40])
    R_max = 40
    V_min = 30
    centers, radii = circ_hough(img, R_max, dim, V_min)

    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    for (x, y), r in zip(centers, radii):
        circle = plt.Circle((x, y), r, color="red", fill=False)
        ax.add_patch(circle)
    ax.set_title("Detected Circles")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    pytest.main([__file__])
