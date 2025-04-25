import numpy as np
import pytest

from hist_utils import calculate_hist_of_img

# -----------------------------------------------------------------------------
# Basic Functionality Tests
# -----------------------------------------------------------------------------


def test_basic_histogram_counts():
    """
    Test correct histogram counts on a small 2x2 grayscale image.
    The expected histogram should map pixel values to their counts.
    """
    img = np.array(
        [[0.0, 0.5], [0.5, 1.0]]
    )  # Image with pixel values 0.0, 0.5, 1.0
    expected = {0.0: 1, 0.5: 2, 1.0: 1}  # Expected histogram counts
    result = calculate_hist_of_img(
        img, return_normalized=False
    )  # Get histogram (counts)
    assert (
        result == expected
    )  # Assert that the result matches the expected counts


def test_normalized_histogram():
    """
    Test normalized histogram. The sum of the histogram should be 1,
    and each value should reflect the proportion of pixels in the image.
    """
    img = np.array(
        [[0.0, 0.5], [0.5, 1.0]]
    )  # Image with pixel values 0.0, 0.5, 1.0
    expected = {
        0.0: 0.25,
        0.5: 0.5,
        1.0: 0.25,
    }  # Expected normalized histogram
    result = calculate_hist_of_img(
        img, return_normalized=True
    )  # Get normalized histogram
    for k in expected:
        # Assert that each key in the expected matches the normalized result
        assert np.isclose(result[k], expected[k]), f"Mismatch for key {k}"


def test_all_same_pixel():
    """
    Test histogram of an image where all pixels have the same value.
    The expected counts should reflect the uniformity in pixel values.
    """
    img = np.ones((4, 4)) * 0.25  # Image where all pixels are 0.25
    expected_counts = {0.25: 16}  # Expected counts for pixel value 0.25
    expected_probs = {
        0.25: 1.0
    }  # Expected normalized probability for pixel value 0.25

    # Calculate both counts and normalized probabilities
    result_counts = calculate_hist_of_img(img, return_normalized=False)
    result_probs = calculate_hist_of_img(img, return_normalized=True)

    assert (
        result_counts == expected_counts
    )  # Assert that the counts match the expected
    assert np.isclose(
        result_probs[0.25], expected_probs[0.25]
    )  # Assert normalized values


def test_empty_image():
    """
    Test behavior when the image is empty.
    The function should return an empty histogram in both cases
    (counts and normalized).
    """
    img = np.array([]).reshape(0, 0)  # Empty image (shape 0x0)
    result = calculate_hist_of_img(img, return_normalized=False)
    assert result == {}  # The result should be an empty dictionary for counts

    result_norm = calculate_hist_of_img(img, return_normalized=True)
    assert (
        result_norm == {}
    )  # The result should be an empty dictionary for normalized values


# -----------------------------------------------------------------------------
# Input Validation Tests
# -----------------------------------------------------------------------------


def test_invalid_input_shape():
    """
    Test that a non-2D image (e.g., RGB image) raises a ValueError.
    The function expects a 2D grayscale array.
    """
    img = np.zeros((2, 2, 3))  # RGB image (invalid input)
    with pytest.raises(
        ValueError, match="Input image must be a 2D grayscale array"
    ):
        calculate_hist_of_img(img)  # Should raise ValueError


def test_pixel_value_range_check():
    """
    Test that pixel values outside the valid range [0, 1] raise a ValueError.
    """
    img = np.array(
        [[0.0, 1.1]]
    )  # Image with pixel value 1.1, which is outside the range
    with pytest.raises(
        ValueError, match="Pixel values must be between 0 and 1"
    ):
        calculate_hist_of_img(img)  # Should raise ValueError


# -----------------------------------------------------------------------------
# Precision and Edge Case Tests
# -----------------------------------------------------------------------------


def test_rounding_precision_effect():
    """
    Check behavior for very close values that are not rounded
    in the function. Should treat values like 0.50001 and 0.49999
    as distinct keys in the histogram.
    """
    img = np.array([[0.50001, 0.49999]])
    expected = {0.50001: 1, 0.49999: 1}  # Expected histogram counts
    result = calculate_hist_of_img(img, return_normalized=False)
    assert result == expected, "Histogram does not match expected counts"
