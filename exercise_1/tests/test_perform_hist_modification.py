import numpy as np
import pytest

from hist_modif import perform_hist_modification
from hist_utils import calculate_hist_of_img, show_histogram


def test_perform_hist_modification_identity():
    """
    Test that modifying an image using its own histogram gives
    similar image.
    """
    img = np.array(
        [[0.0, 0.0, 0.5], [0.5, 0.5, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64
    )

    # Calculate original histogram
    hist = calculate_hist_of_img(img, return_normalized=True)
    show_histogram(hist, title="Original Histogram")
    # bin_centers = np.linspace(0, 1, 256)[1:] - 1/512  # 255 centers

    # Perform histogram modification with self
    modified_img = perform_hist_modification(img, hist_ref=hist, mode="greedy")
    modified_hist = calculate_hist_of_img(modified_img, return_normalized=True)
    show_histogram(modified_hist, title="Modified Histogram")

    # Since we're using the same histogram, the result should be similar
    assert np.allclose(
        modified_img, img, atol=1e-2
    ), "Identity histogram should preserve image"


def test_perform_hist_modification_stretch():
    """
    Test that histogram modification stretches contrast
    when hist_ref is different.
    """
    img = np.ones((4, 4), dtype=np.float32) * 0.5  # All pixels are 0.5

    # Target histogram is a ramp from dark to bright
    hist_ref = {
        round(i / 255.0, 3): 1 for i in range(256)
    }  # Uniform distribution

    modified_img = perform_hist_modification(
        img, hist_ref=hist_ref, mode="greedy"
    )

    # All pixels originally 0.5, should now be mapped to somewhere in middle
    #  of the target
    assert not np.allclose(
        modified_img, img
    ), "Image should be transformed under different histogram"
    assert (
        modified_img.min() >= 0.0 and modified_img.max() <= 1.0
    ), "Output image values should remain in [0,1]"


def test_perform_hist_modification_bimodal():
    """Test with an image that has two levels: dark and bright."""
    img = np.array(
        [[0.2, 0.2, 0.2], [0.8, 0.8, 0.8], [0.2, 0.8, 0.2]], dtype=np.float32
    )

    # Reference histogram prefers lower values
    hist_ref = {0.1: 5, 0.9: 1}

    modified_img = perform_hist_modification(
        img, hist_ref=hist_ref, mode="greedy"
    )

    # Expect 0.2s to stay low, and 0.8s to shift closer to 0.9
    assert modified_img.shape == img.shape
    assert modified_img.min() >= 0.0 and modified_img.max() <= 1.0

    # Check that mapped values follow expected trend
    assert (
        modified_img[0, 0] < modified_img[1, 0]
    ), "Dark values should remain darker than bright ones"


def test_invalid_input_type():
    """Ensure invalid input types raise proper errors."""
    with pytest.raises(TypeError):
        perform_hist_modification(
            np.ones((4, 4), dtype=np.uint8), hist_ref={}, mode="greedy"
        )


def test_invalid_mode():
    """Ensure unsupported modes raise error."""
    img = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(NotImplementedError):
        perform_hist_modification(img, hist_ref={1.0: 1.0}, mode="non-greedy")


def test_invalid_hist_ref_range():
    """Ensure invalid gray level keys raise error."""
    img = np.ones((4, 4), dtype=np.float32)
    hist_ref = {1.2: 1.0}  # Invalid key (> 1)
    with pytest.raises(ValueError):
        perform_hist_modification(img, hist_ref=hist_ref, mode="greedy")
