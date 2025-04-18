import numpy as np
import pytest

from hist_utils import calculate_hist_of_img

# -----------------------------------------------------------------------------
# Basic Functionality Tests
# -----------------------------------------------------------------------------


def test_basic_histogram_counts():
    """Test correct histogram counts on a small image."""
    img = np.array([[0.0, 0.5], [0.5, 1.0]])
    expected = {0.0: 1, 0.5: 2, 1.0: 1}
    result = calculate_hist_of_img(img, return_normalized=False)
    assert result == expected


def test_normalized_histogram():
    """Test normalized histogram sums to 1 and has correct proportions."""
    img = np.array([[0.0, 0.5], [0.5, 1.0]])
    expected = {0.0: 0.25, 0.5: 0.5, 1.0: 0.25}
    result = calculate_hist_of_img(img, return_normalized=True)
    for k in expected:
        assert np.isclose(result[k], expected[k]), f"Mismatch for key {k}"


def test_all_same_pixel():
    """Test histogram of an image where all pixels have the same value."""
    img = np.ones((4, 4)) * 0.25
    expected_counts = {0.25: 16}
    expected_probs = {0.25: 1.0}

    result_counts = calculate_hist_of_img(img, return_normalized=False)
    result_probs = calculate_hist_of_img(img, return_normalized=True)

    assert result_counts == expected_counts
    assert np.isclose(result_probs[0.25], expected_probs[0.25])


def test_empty_image():
    """Test that an empty image returns an empty histogram."""
    img = np.array([]).reshape(0, 0)
    result = calculate_hist_of_img(img, return_normalized=False)
    assert result == {}

    result_norm = calculate_hist_of_img(img, return_normalized=True)
    assert result_norm == {}


# -----------------------------------------------------------------------------
# Input Validation Tests
# -----------------------------------------------------------------------------


def test_invalid_input_shape():
    """Test that a non-2D image raises a ValueError."""
    img = np.zeros((2, 2, 3))  # RGB image
    with pytest.raises(
        ValueError, match="Input image must be a 2D grayscale array"
    ):
        calculate_hist_of_img(img)


def test_pixel_value_range_check():
    """Test that pixel values outside [0, 1] raise a ValueError."""
    img = np.array([[0.0, 1.1]])
    with pytest.raises(
        ValueError, match="Pixel values must be between 0 and 1"
    ):
        calculate_hist_of_img(img)


# -----------------------------------------------------------------------------
# Precision and Edge Case Tests
# -----------------------------------------------------------------------------


def test_rounding_precision_effect():
    """
    Check behavior for very close values that are not rounded in the function.
    Should treat 0.50001 and 0.49999 as distinct keys.
    """
    img = np.array([[0.50001, 0.49999]])
    result = calculate_hist_of_img(img, return_normalized=False)
    assert result == {0.50001: 1, 0.49999: 1}


def test_fine_grained_float_values():
    """
    Test that histogram handles finely spaced float values correctly.
    """
    img = np.linspace(0.0, 1.0, 10).reshape(2, 5)
    result = calculate_hist_of_img(img, return_normalized=False)
    assert len(result) == 10
    assert all(v == 1 for v in result.values())


def test_image_with_zeros():
    """Test an image that contains only zero-valued pixels."""
    img = np.zeros((8, 8))
    result = calculate_hist_of_img(img, return_normalized=False)
    assert result == {0.0: 64}

    result_norm = calculate_hist_of_img(img, return_normalized=True)
    assert np.isclose(result_norm[0.0], 1.0)
