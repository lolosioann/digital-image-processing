import numpy as np
import pytest

from part2.fir_conv import fir_conv


def test_identity_kernel():
    img = np.random.rand(5, 5)
    kernel = np.zeros((3, 3))
    kernel[1, 1] = 1  # Identity kernel
    out_img, out_origin = fir_conv(img, kernel)

    assert np.allclose(out_img, img), "Identity kernel should return the original image"
    assert np.array_equal(
        out_origin, np.array([0, 0])
    ), "Default origin should be (0,0)"


def test_custom_origin():
    img = np.ones((5, 5))
    kernel = np.ones((3, 3)) / 9
    in_origin = np.array([2, 2])
    mask_origin = np.array([1, 1])  # Center of the kernel
    _, out_origin = fir_conv(img, kernel, in_origin, mask_origin)

    expected_origin = in_origin + mask_origin - np.array([1, 1])  # Since pad = (1,1)
    assert np.array_equal(
        out_origin, expected_origin
    ), f"Expected {expected_origin}, got {out_origin}"


def test_non_centered_mask_origin():
    img = np.ones((5, 5))
    kernel = np.ones((3, 3)) / 9
    in_origin = np.array([1, 1])
    mask_origin = np.array([0, 0])  # Top-left of kernel
    _, out_origin = fir_conv(img, kernel, in_origin, mask_origin)

    expected_origin = in_origin + mask_origin - np.array([1, 1])
    assert np.array_equal(
        out_origin, expected_origin
    ), f"Expected {expected_origin}, got {out_origin}"


def test_shape_preservation():
    img = np.random.rand(10, 15)
    kernel = np.ones((5, 3)) / 15
    out_img, _ = fir_conv(img, kernel)

    assert out_img.shape == img.shape, "Output image shape must match input image shape"


def test_input_validation():
    with pytest.raises(ValueError):
        fir_conv(None, np.ones((3, 3)))

    with pytest.raises(ValueError):
        fir_conv(np.random.rand(5, 5, 3), np.ones((3, 3)))  # Not a 2D image

    with pytest.raises(ValueError):
        fir_conv(np.random.rand(5, 5) * 2, np.ones((3, 3)))  # Values > 1

    with pytest.raises(ValueError):
        fir_conv(np.random.rand(5, 5), None)
