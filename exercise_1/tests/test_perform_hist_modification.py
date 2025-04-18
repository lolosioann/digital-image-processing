import numpy as np
import pytest

from hist_modif import perform_hist_modification
from hist_utils import calculate_hist_of_img


class TestHistogramModification:
    def test_identity_mapping_preserves_image(self):
        """
        Modifying an image using its own histogram should
        yield a similar result.
        """
        img = np.array(
            [[0.0, 0.0, 0.5], [0.5, 0.5, 1.0], [1.0, 1.0, 1.0]],
            dtype=np.float64,
        )

        hist = calculate_hist_of_img(img, return_normalized=True)

        modified_img = perform_hist_modification(
            img, hist_ref=hist, mode="greedy"
        )

        assert np.allclose(
            modified_img, img, atol=1e-2
        ), "Image should remain unchanged when using its own histogram"

    def test_contrast_stretching_with_uniform_hist(self):
        """
        A uniform target histogram should stretch the dynamic range.
        """
        img = np.ones((4, 4), dtype=np.float32) * 0.5
        hist_ref = {round(i / 255.0, 3): 1 for i in range(256)}

        modified_img = perform_hist_modification(
            img, hist_ref=hist_ref, mode="greedy"
        )

        assert not np.allclose(
            modified_img, img
        ), "Uniform histogram should transform a constant image"
        assert (
            0.0 <= modified_img.min() <= 1.0
            and 0.0 <= modified_img.max() <= 1.0
        )

    def test_bimodal_distribution_mapping(self):
        """
        Dark and bright values should map according to the
        skewed target distribution.
        """
        img = np.array(
            [[0.2, 0.2, 0.2], [0.8, 0.8, 0.8], [0.2, 0.8, 0.2]],
            dtype=np.float32,
        )
        hist_ref = {0.1: 5, 0.9: 1}

        modified_img = perform_hist_modification(
            img, hist_ref=hist_ref, mode="greedy"
        )

        assert modified_img.shape == img.shape
        assert (
            0.0 <= modified_img.min() <= 1.0
            and 0.0 <= modified_img.max() <= 1.0
        )
        assert (
            modified_img[0, 0] < modified_img[1, 0]
        ), "Darker values should map to lower intensities"

    @pytest.mark.parametrize(
        "img, hist_ref, mode, expected_exception",
        [
            (np.ones((4, 4), dtype=np.uint8), {}, "greedy", TypeError),
            (
                np.ones((4, 4), dtype=np.float32),
                {1.2: 1.0},
                "greedy",
                ValueError,
            ),
            (
                np.ones((4, 4), dtype=np.float32),
                {1.0: 1.0},
                "non-greedy",
                NotImplementedError,
            ),
        ],
    )
    def test_invalid_inputs(self, img, hist_ref, mode, expected_exception):
        """
        Parameterized test for invalid input scenarios.
        """
        with pytest.raises(expected_exception):
            perform_hist_modification(img, hist_ref=hist_ref, mode=mode)
