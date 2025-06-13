import numpy as np
import pytest

from image_to_graph import (
    image_to_graph,  # Replace with the actual module name
)


def test_affinity_matrix_shape():
    img = np.random.rand(4, 4, 3)  # Small RGB image
    affinity = image_to_graph(img)
    assert affinity.shape == (
        16,
        16,
    ), "Affinity matrix shape should be (M*N, M*N)"


def test_affinity_matrix_symmetry():
    img = np.random.rand(3, 3, 1)  # Grayscale image
    affinity = image_to_graph(img)
    assert np.allclose(
        affinity, affinity.T
    ), "Affinity matrix must be symmetric"


def test_affinity_diagonal_values():
    img = np.random.rand(2, 2, 1)
    affinity = image_to_graph(img)
    diag = np.diag(affinity)
    # d(i,i) = 0 -> A(i,i) = 1/exp(0) = 1
    assert np.allclose(diag, np.ones_like(diag)), "Diagonal values should be 1"


def test_affinity_value_range():
    img = np.random.rand(3, 3, 2)
    affinity = image_to_graph(img)
    assert np.all(affinity > 0) and np.all(
        affinity <= 1
    ), "All affinities should be in (0, 1]"


def test_affinity_identical_pixels():
    img = np.ones((2, 2, 3)) * 0.5
    affinity = image_to_graph(img)
    # All pixels are the same, so distance = 0 => affinity = 1
    assert np.allclose(
        affinity, np.ones((4, 4))
    ), "If all pixels are identical, affinity should be 1 everywhere"
