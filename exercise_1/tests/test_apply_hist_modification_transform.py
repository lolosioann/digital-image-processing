import numpy as np
import pytest

from hist_utils import apply_hist_modification_transform


class TestHistogramTransform:
    def test_basic_transformation(self):
        img = np.array([[0.0, 0.5], [0.5, 1.0]])
        transform = {0.0: 0.1, 0.5: 0.6, 1.0: 0.9}
        expected = np.array([[0.1, 0.6], [0.6, 0.9]])
        result = apply_hist_modification_transform(img, transform)
        assert np.allclose(result, expected)

    def test_missing_mapping_raises_error(self):
        img = np.array([[0.1, 0.3]])
        transform = {0.1: 0.2}  # 0.3 is missing
        with pytest.raises(
            ValueError,
            match="Level 0.3 does not exist in the modification transform",
        ):
            apply_hist_modification_transform(img, transform)

    def test_input_not_2d(self):
        img = np.array([0.0, 0.5, 1.0])  # 1D array
        transform = {0.0: 0.1, 0.5: 0.6, 1.0: 0.9}
        with pytest.raises(
            ValueError, match="Input image must be a 2D grayscale array"
        ):
            apply_hist_modification_transform(img, transform)

    def test_out_of_range_values(self):
        img = np.array([[0.0, 1.2]])  # 1.2 is invalid
        transform = {0.0: 0.1, 1.0: 0.9, 1.2: 1.0}
        with pytest.raises(
            ValueError, match="Pixel values must be between 0 and 1"
        ):
            apply_hist_modification_transform(img, transform)

    def test_all_pixels_same_value(self):
        img = np.full((5, 5), 0.25)
        transform = {0.25: 0.75}
        result = apply_hist_modification_transform(img, transform)
        expected = np.full((5, 5), 0.75)
        assert np.allclose(result, expected)

    def test_empty_image(self):
        img = np.array([]).reshape(0, 0)
        transform = {}
        result = apply_hist_modification_transform(img, transform)
        assert result.size == 0
        assert result.shape == (0, 0)
