import numpy as np
import pytest
from hist_utils import calculate_hist_of_img  # Replace with actual module name


def test_basic_histogram_counts():
    img = np.array([[0.0, 0.5], [0.5, 1.0]])
    expected = {0.0: 1, 0.5: 2, 1.0: 1}
    result = calculate_hist_of_img(img, return_normalized=False)
    assert result == expected


def test_normalized_histogram():
    img = np.array([[0.0, 0.5], [0.5, 1.0]])
    expected = {0.0: 0.25, 0.5: 0.5, 1.0: 0.25}
    result = calculate_hist_of_img(img, return_normalized=True)
    for k in expected:
        assert np.isclose(result[k], expected[k]), f"Mismatch for key {k}"


def test_rounding_precision_effect():
    img = np.array([[0.50001, 0.49999]])
    result = calculate_hist_of_img(img, return_normalized=False)
    # Expect both values to round to 0.5 and be grouped
    assert result == {0.50001: 1, 0.49999: 1}


def test_invalid_input_shape():
    img = np.zeros((2, 2, 3))  # Not 2D
    with pytest.raises(ValueError, match="Input image must be a 2D grayscale array"):
        calculate_hist_of_img(img)


def test_pixel_value_range_check():
    img = np.array([[0.0, 1.1]])
    with pytest.raises(ValueError, match="Pixel values must be between 0 and 1"):
        calculate_hist_of_img(img)


def test_all_same_pixel():
    img = np.ones((4, 4)) * 0.25
    result = calculate_hist_of_img(img, return_normalized=False)
    assert result == {0.25: 16}

    result_norm = calculate_hist_of_img(img, return_normalized=True)
    assert np.isclose(result_norm[0.25], 1.0)


def test_empty_image():
    img = np.array([]).reshape(0, 0)
    result = calculate_hist_of_img(img, return_normalized=False)
    assert result == {}

    result_norm = calculate_hist_of_img(img, return_normalized=True)
    assert result_norm == {}
