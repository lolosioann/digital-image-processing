import numpy as np
import pytest

from hist_utils import apply_hist_modification_transform


class TestHistogramTransform:
    def test_basic_transformation(self):
        """
        Test case for a basic transformation where the pixel values are mapped
        according to the provided dictionary. The result should be the image
        with transformed pixel values.
        """
        img = np.array([[0.0, 0.5], [0.5, 1.0]])  # Sample image array
        transform = {0.0: 0.1, 0.5: 0.6, 1.0: 0.9}  # Transformation mapping
        expected = np.array([[0.1, 0.6], [0.6, 0.9]])  # Expected result
        result = apply_hist_modification_transform(img, transform)
        assert np.allclose(
            result, expected
        )  # Assert if transformation is correct

    def test_missing_mapping_raises_error(self):
        """
        Test case where a pixel value in the image does not have a
        corresponding mapping in the transformation dictionary.
        This should raise a ValueError.
        """
        img = np.array([[0.1, 0.3]])  # Image with values 0.1 and 0.3
        transform = {0.1: 0.2}  # Missing mapping for 0.3
        with pytest.raises(
            ValueError,
            match="Level 0.3 does not exist in the modification transform",
        ):
            apply_hist_modification_transform(img, transform)

    def test_input_not_2d(self):
        """
        Test case for handling non-2D input arrays. The function should raise a
        ValueError if the input image is not 2D.
        """
        img = np.array([0.0, 0.5, 1.0])  # 1D image (invalid input)
        transform = {0.0: 0.1, 0.5: 0.6, 1.0: 0.9}
        with pytest.raises(
            ValueError,
            match="Input image must be a 2D grayscale array",
        ):
            apply_hist_modification_transform(img, transform)

    def test_out_of_range_values(self):
        """
        Test case where pixel values in the image are out of the valid
        range [0, 1]. The function should raise a ValueError.
        """
        img = np.array([[0.0, 1.2]])  # Image with an invalid value (1.2)
        transform = {0.0: 0.1, 1.0: 0.9, 1.2: 1.0}
        with pytest.raises(
            ValueError,
            match="Pixel values must be between 0 and 1",
        ):
            apply_hist_modification_transform(img, transform)

    def test_all_pixels_same_value(self):
        """
        Test case where all pixels in the image have the same value.
        The transformation should apply uniformly across the image.
        """
        img = np.full((5, 5), 0.25)  # Image where all pixels are 0.25
        transform = {0.25: 0.75}  # Transformation mapping for 0.25 -> 0.75
        result = apply_hist_modification_transform(img, transform)
        expected = np.full(
            (5, 5), 0.75
        )  # Expected result where all pixels are 0.75
        assert np.allclose(
            result, expected
        )  # Assert if transformation is correct

    def test_empty_image(self):
        """
        Test case for an empty image. The function should handle it
        gracefully and return an empty array.
        """
        img = np.array([]).reshape(0, 0)  # Empty image (shape (0, 0))
        transform = {}  # No transformation (empty dictionary)
        result = apply_hist_modification_transform(img, transform)
        assert result.size == 0  # The result should be an empty array
        assert result.shape == (0, 0)  # The shape should be (0, 0)
