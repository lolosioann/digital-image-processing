import numpy as np
import pytest

from sobel_edge import (
    sobel_edge,  # Replace 'your_module' with the actual module name
)


def test_output_shape_and_dtype():
    img = np.random.rand(100, 100)
    thres = 0.3
    result = sobel_edge(img, thres)
    assert result.shape == img.shape, "Output shape should match input"
    assert (
        result.dtype == int or result.dtype == np.int_
    ), "Output dtype must be int"


def test_binary_output():
    img = np.random.rand(50, 50)
    thres = 0.5
    result = sobel_edge(img, thres)
    unique_vals = np.unique(result)
    assert np.all(
        np.isin(unique_vals, [0, 1])
    ), "Output must be binary (0 or 1)"


def test_invalid_image_shape():
    with pytest.raises(ValueError):
        sobel_edge(np.random.rand(10, 10, 3), thres=0.5)


def test_invalid_image_range():
    img = np.random.uniform(-1, 2, size=(10, 10))
    with pytest.raises(ValueError):
        sobel_edge(img, thres=0.5)


def test_invalid_threshold():
    img = np.random.rand(10, 10)
    with pytest.raises(ValueError):
        sobel_edge(img, thres=0)


def test_known_pattern():
    # Create a simple test pattern: left half 0, right half 1 (vertical edge)
    img = np.zeros((5, 5))
    img[:, 2:] = 1.0
    result = sobel_edge(img, thres=0.5)
    # We expect edges to appear around the vertical transition at column 2
    edge_columns = result[:, 1:4].sum(axis=1)  # Check columns around edge
    assert np.all(edge_columns > 0), "Expected edge at vertical transition"


def test_no_edges_for_uniform_image():
    img = np.ones((20, 20)) * 0.7
    result = sobel_edge(img, thres=0.1)
    assert np.all(result == 0), "Uniform image should have no edges"
