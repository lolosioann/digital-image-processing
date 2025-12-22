import numpy as np
import pytest

from part2.log_edge import log_edge


def test_output_shape_and_type():
    img = np.random.rand(64, 64)
    result = log_edge(img, sigma=1.0)
    assert result.shape == img.shape
    assert result.dtype == int or result.dtype == np.int_


def test_binary_output_values():
    img = np.random.rand(32, 32)
    result = log_edge(img, sigma=1.0)
    unique = np.unique(result)
    assert np.all(np.isin(unique, [0, 1]))


def test_invalid_input_shape():
    with pytest.raises(ValueError):
        log_edge(np.random.rand(32, 32, 3), sigma=1.0)


def test_input_range_check():
    img = np.random.uniform(-0.5, 1.5, size=(10, 10))
    with pytest.raises(ValueError):
        log_edge(img, sigma=1.0)


def test_invalid_sigma():
    img = np.random.rand(10, 10)
    with pytest.raises(ValueError):
        log_edge(img, sigma=-1)


def test_zero_crossings_on_known_edge():
    img = np.zeros((10, 10))
    img[:, 5:] = 1.0  # Κάθετη ακμή
    result = log_edge(img, sigma=1.0)
    # Αναμένουμε ακμές γύρω από τη στήλη 5
    assert np.sum(result[:, 4:6]) > 0


# def test_no_edges_for_uniform_image():
#     img = np.ones((30, 30)) * 0.7
#     result = log_edge(img, sigma=1.0)
#     assert np.all(result == 0)
