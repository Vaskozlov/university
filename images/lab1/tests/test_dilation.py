"""Tests for the dilation implementations and supporting modules."""

from __future__ import annotations

import numpy as np
import pytest

from dilation_lab.benchmark import run_benchmark
from dilation_lab.dilation import dilate_opencv, dilate_python
from dilation_lab.io_utils import binarize
from dilation_lab.validation import compare_results


@pytest.mark.parametrize("dilate", [dilate_python, dilate_opencv], ids=["python", "opencv"])
def test_all_black(dilate):
    image = np.zeros((7, 9), dtype=np.uint8)
    assert np.array_equal(dilate(image), image)


@pytest.mark.parametrize("dilate", [dilate_python, dilate_opencv], ids=["python", "opencv"])
def test_all_white(dilate):
    image = np.full((7, 9), 255, dtype=np.uint8)
    assert np.array_equal(dilate(image), image)


@pytest.mark.parametrize("dilate", [dilate_python, dilate_opencv], ids=["python", "opencv"])
def test_center_point_becomes_3x3(dilate):
    image = np.zeros((5, 5), dtype=np.uint8)
    image[2, 2] = 255
    expected = np.zeros((5, 5), dtype=np.uint8)
    expected[1:4, 1:4] = 255
    assert np.array_equal(dilate(image), expected)


@pytest.mark.parametrize("dilate", [dilate_python, dilate_opencv], ids=["python", "opencv"])
def test_corner_point_becomes_2x2(dilate):
    image = np.zeros((5, 5), dtype=np.uint8)
    image[0, 0] = 255
    expected = np.zeros((5, 5), dtype=np.uint8)
    expected[0:2, 0:2] = 255
    assert np.array_equal(dilate(image), expected)


@pytest.mark.parametrize("dilate", [dilate_python, dilate_opencv], ids=["python", "opencv"])
def test_edge_point(dilate):
    image = np.zeros((5, 5), dtype=np.uint8)
    image[0, 2] = 255
    expected = np.zeros((5, 5), dtype=np.uint8)
    expected[0:2, 1:4] = 255
    assert np.array_equal(dilate(image), expected)


@pytest.mark.parametrize("dilate", [dilate_python, dilate_opencv], ids=["python", "opencv"])
def test_diagonal_points(dilate):
    image = np.zeros((5, 5), dtype=np.uint8)
    image[1, 1] = 255
    image[3, 3] = 255
    expected = np.zeros((5, 5), dtype=np.uint8)
    expected[0:3, 0:3] = 255
    expected[2:5, 2:5] = 255
    assert np.array_equal(dilate(image), expected)


@pytest.mark.parametrize("dilate", [dilate_python, dilate_opencv], ids=["python", "opencv"])
def test_diagonal_neighbors_count(dilate):
    # A single center pixel must whiten all eight neighbors, including
    # the four diagonal ones.
    image = np.zeros((3, 3), dtype=np.uint8)
    image[1, 1] = 255
    assert np.array_equal(dilate(image), np.full((3, 3), 255, dtype=np.uint8))


@pytest.mark.parametrize("dilate", [dilate_python, dilate_opencv], ids=["python", "opencv"])
@pytest.mark.parametrize("value", [0, 255])
def test_single_pixel(dilate, value):
    image = np.full((1, 1), value, dtype=np.uint8)
    assert np.array_equal(dilate(image), image)


@pytest.mark.parametrize("dilate", [dilate_python, dilate_opencv], ids=["python", "opencv"])
def test_narrow_row(dilate):
    image = np.array([[0, 255, 0, 0, 255]], dtype=np.uint8)
    expected = np.full((1, 5), 255, dtype=np.uint8)
    assert np.array_equal(dilate(image), expected)


@pytest.mark.parametrize("dilate", [dilate_python, dilate_opencv], ids=["python", "opencv"])
def test_narrow_column(dilate):
    image = np.array([[0], [255], [0], [0], [255]], dtype=np.uint8)
    expected = np.full((5, 1), 255, dtype=np.uint8)
    assert np.array_equal(dilate(image), expected)


def test_random_masks_match_opencv():
    rng = np.random.default_rng(42)
    for size in (8, 17, 32, 64):
        white = rng.random((size, size)) < 0.25
        image = (white * 255).astype(np.uint8)
        input_before = image.copy()

        result_python = dilate_python(image)
        result_opencv = dilate_opencv(image)

        assert result_python.dtype == np.uint8
        assert result_python.shape == image.shape
        assert np.array_equal(result_python, result_opencv)
        assert np.array_equal(image, input_before)


def test_input_not_mutated():
    image = np.zeros((6, 6), dtype=np.uint8)
    image[3, 3] = 255
    before = image.copy()
    dilate_python(image)
    dilate_opencv(image)
    assert np.array_equal(image, before)


def test_returns_new_array():
    image = np.zeros((4, 4), dtype=np.uint8)
    image[1, 1] = 255
    assert dilate_python(image) is not image
    assert dilate_opencv(image) is not image


def test_output_dtype_and_shape():
    image = np.zeros((5, 7), dtype=np.uint8)
    image[2, 3] = 255
    result = dilate_python(image)
    assert result.dtype == np.uint8
    assert result.shape == image.shape


def test_compare_results_identical():
    image = np.zeros((5, 5), dtype=np.uint8)
    image[2, 2] = 255
    report = compare_results(dilate_python(image), dilate_opencv(image))
    assert report["identical"] is True
    assert report["mismatched_pixels"] == 0
    assert report["max_abs_difference"] == 0
    assert np.array_equal(report["difference"], np.zeros((5, 5), dtype=np.uint8))


def test_compare_results_mismatch():
    a = np.zeros((3, 3), dtype=np.uint8)
    b = np.zeros((3, 3), dtype=np.uint8)
    b[0, 0] = 255
    report = compare_results(a, b)
    assert report["identical"] is False
    assert report["mismatched_pixels"] == 1
    assert report["max_abs_difference"] == 255
    assert report["difference"][0, 0] == 255


def test_max_abs_difference_without_uint8_overflow():
    # 0 - 255 would wrap to 1 in uint8 arithmetic; the true difference is 255.
    a = np.array([[0]], dtype=np.uint8)
    b = np.array([[255]], dtype=np.uint8)
    report = compare_results(a, b)
    assert report["max_abs_difference"] == 255


def test_compare_results_shape_mismatch():
    a = np.zeros((3, 3), dtype=np.uint8)
    b = np.zeros((4, 4), dtype=np.uint8)
    report = compare_results(a, b)
    assert report["identical"] is False
    assert report["mismatched_pixels"] == -1
    assert report["max_abs_difference"] == -1


def test_binarize_threshold_validation():
    gray = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        binarize(gray, -1)
    with pytest.raises(ValueError):
        binarize(gray, 256)


def test_binarize_strictly_greater():
    gray = np.array([[126, 127, 128]], dtype=np.uint8)
    binary = binarize(gray, 127)
    assert np.array_equal(binary, np.array([[0, 0, 255]], dtype=np.uint8))


def test_benchmark_rejects_non_positive_sizes():
    with pytest.raises(ValueError, match="sizes"):
        run_benchmark([0])
    with pytest.raises(ValueError, match="sizes"):
        run_benchmark([64, 0, 128])


def test_binarize_produces_only_zero_and_255():
    gray = np.random.default_rng(7).integers(0, 256, size=(32, 32), dtype=np.uint8)
    binary = binarize(gray, 127)
    assert binary.dtype == np.uint8
    assert set(np.unique(binary)) <= {0, 255}
    assert np.array_equal(binary, (gray > 127).astype(np.uint8) * 255)
