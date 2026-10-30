"""Pixel-by-pixel comparison of the two dilation results."""

from __future__ import annotations

import numpy as np


def compare_results(
    python_result: np.ndarray, opencv_result: np.ndarray
) -> dict[str, object]:
    """Compare two dilation results.

    Args:
        python_result: Result of the native Python implementation.
        opencv_result: Result of the OpenCV implementation.

    Returns:
        Dictionary with keys:

        - ``shape_match`` (bool): shapes are equal.
        - ``dtype_match`` (bool): dtypes are equal.
        - ``identical`` (bool): ``np.array_equal`` on both results.
        - ``mismatched_pixels`` (int): count of differing pixels,
          or -1 if the shapes differ.
        - ``max_abs_difference`` (int): maximum absolute pixel
          difference computed without uint8 overflow, or -1 if the
          shapes differ.
        - ``difference`` (np.ndarray): 255 where the results differ,
          0 elsewhere.
    """
    shape_match = python_result.shape == opencv_result.shape
    dtype_match = python_result.dtype == opencv_result.dtype

    if shape_match:
        # Cast before subtracting so uint8 subtraction cannot wrap.
        diff = python_result.astype(np.int16) - opencv_result.astype(np.int16)
        max_abs_difference = int(np.max(np.abs(diff))) if diff.size else 0
        mismatch_mask = python_result != opencv_result
        mismatched_pixels = int(np.count_nonzero(mismatch_mask))
        difference = np.where(mismatch_mask, 255, 0).astype(np.uint8)
    else:
        max_abs_difference = -1
        mismatched_pixels = -1
        difference = np.zeros(python_result.shape, dtype=np.uint8)

    return {
        "shape_match": shape_match,
        "dtype_match": dtype_match,
        "identical": bool(np.array_equal(python_result, opencv_result)),
        "mismatched_pixels": mismatched_pixels,
        "max_abs_difference": max_abs_difference,
        "difference": difference,
    }
