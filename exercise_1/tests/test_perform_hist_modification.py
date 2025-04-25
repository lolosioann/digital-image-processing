import numpy as np
import pytest

from hist_modif import perform_hist_modification
from hist_utils import calculate_hist_of_img


class TestHistogramModification:
    # ----------------------------------------------------------
    # Identity Mapping Test
    # ----------------------------------------------------------
    def test_identity_mapping_preserves_image(self):
        """
        Test that modifying an image using its own histogram should
        yield a similar result. This ensures that no unintended
        transformations occur when the image is modified
        with its own histogram.
        """
        # Create a simple image for testing
        img = np.array(
            [[0.0, 0.0, 0.5], [0.5, 0.5, 1.0], [1.0, 1.0, 1.0]],
            dtype=np.float64,
        )

        # Calculate the histogram of the image
        hist = calculate_hist_of_img(img, return_normalized=True)

        # Perform histogram modification with the calculated
        # histogram
        modified_img = perform_hist_modification(
            img, hist_ref=hist, mode="greedy"
        )

        # Assert that the modified image is very close to the
        # original image
        assert np.allclose(
            modified_img, img, atol=1e-2
        ), "Image should remain unchanged when using its own "
        "histogram"

    # ----------------------------------------------------------
    # Contrast Stretching Test with Uniform Histogram
    # ----------------------------------------------------------
    def test_contrast_stretching_with_uniform_hist(self):
        """
        Test that a uniform target histogram should stretch the
        dynamic range. This ensures that a constant image will be
        transformed and the dynamic range of the modified image will
        expand.
        """
        # Create a constant image with a single pixel value (0.5)
        img = np.ones((4, 4), dtype=np.float32) * 0.5

        # Define a uniform target histogram
        hist_ref = {round(i / 255.0, 3): 1.0 for i in range(256)}

        # Perform histogram modification
        modified_img = perform_hist_modification(
            img, hist_ref=hist_ref, mode="greedy"
        )

        # Check if the modified image is different from the original
        assert not np.allclose(
            modified_img, img
        ), "Uniform histogram should transform a constant image"

        # Ensure the pixel values are within the valid range [0, 1]
        assert (
            0.0 <= modified_img.min() <= 1.0
            and 0.0 <= modified_img.max() <= 1.0
        )

    # -------------------------------------------------------
    # Bimodal Distribution Mapping Test
    # -------------------------------------------------------
    def test_bimodal_distribution_mapping(self):
        """
        Test that dark and bright values should map according to the
        skewed target distribution. This checks if the image with a
        bimodal distribution maps correctly to a target with a
        similar distribution.
        """
        # Create an image with two distinct pixel groups: dark and bright
        img = np.array(
            [[0.2, 0.2, 0.2], [0.8, 0.8, 0.8], [0.2, 0.8, 0.2]],
            dtype=np.float32,
        )

        # Define a bimodal target histogram with two dominant values
        hist_ref = {0.1: 3.0 / 9, 0.9: 6.0 / 9}

        # Perform histogram modification
        modified_img = perform_hist_modification(
            img, hist_ref=hist_ref, mode="greedy"
        )

        # Check if the modified image has the same shape as the original
        assert modified_img.shape == img.shape

        # Ensure that the modified image has pixel values within
        # the valid range
        assert (
            0.0 <= modified_img.min() <= 1.0
            and 0.0 <= modified_img.max() <= 1.0
        )

        # Ensure that the modified image contains only two distinct
        # pixel values
        unique_vals, counts = np.unique(modified_img, return_counts=True)
        assert len(unique_vals) == 2

        # Assert that the majority of pixels are mapped to the dominant
        # histogram level
        assert any(
            c > 4 for c in counts
        ), "Majority of pixels should be mapped to dominant histogram level"

    # ----------------------------------------------------------
    # Non-Greedy Mapping Test
    # ----------------------------------------------------------
    def test_non_greedy_mapping_distribution(self):
        """
        Test that non-greedy mapping respects bin capacity and applies
        fair distribution. This test ensures that pixel values are spread
        out according to the target histogram.
        """
        # Create an image with 10 pixels each for 0.0, 0.5, and 1.0
        img = np.array(
            [[0.0] * 10 + [0.5] * 10 + [1.0] * 10], dtype=np.float32
        ).reshape(3, 10)

        # Define a uniform target histogram
        hist_ref = {round(i / 255.0, 3): 1.0 for i in range(256)}

        # Perform histogram modification in non-greedy mode
        modified_img = perform_hist_modification(
            img, hist_ref=hist_ref, mode="non-greedy"
        )

        # Ensure the modified image has the same shape as the original
        assert modified_img.shape == img.shape

        # Check that pixel values are within the valid range [0, 1]
        assert np.all((0.0 <= modified_img) & (modified_img <= 1.0))

    # ----------------------------------------------------
    # Invalid Inputs Test (Parameterized)
    # ----------------------------------------------------
    @pytest.mark.parametrize(
        "img, hist_ref, mode, expected_exception",
        [
            (np.ones((4, 4), dtype=np.uint8), {0.5: 1.0}, "greedy", TypeError),
            (
                np.ones((4, 4), dtype=np.float32),
                {1.2: 1.0},
                "greedy",
                ValueError,
            ),
            (
                np.ones((4, 4), dtype=np.float32),
                {1.0: 1.0},
                "post-disturbance",
                ValueError,
            ),
            (
                np.ones((4, 4), dtype=np.float32),
                {1.0: 1.0},
                "invalid-mode",
                NotImplementedError,
            ),
            (np.ones((4, 4), dtype=np.float32), {}, "greedy", ValueError),
        ],
    )
    def test_invalid_inputs(self, img, hist_ref, mode, expected_exception):
        """
        Parameterized test for invalid input scenarios.
        This test ensures that various invalid inputs result in the appropriate
        exceptions being raised.
        """
        # Assert that the function raises the expected exception
        with pytest.raises(expected_exception):
            perform_hist_modification(img, hist_ref=hist_ref, mode=mode)
