"""Two implementations of 3x3 binary image dilation."""

from __future__ import annotations

import cv2
import numpy as np

#: 3x3 structuring element with all nine positions active.
KERNEL_3X3 = np.ones((3, 3), dtype=np.uint8)

_NEIGHBOR_OFFSETS = (-1, 0, 1)


def dilate_python(image: np.ndarray) -> np.ndarray:
    """Dilate a binary image using explicit Python loops.

    For every output pixel the 3x3 neighborhood is scanned; the output
    is 255 if any existing neighbor is white, otherwise 0. Out-of-bounds
    positions count as 0 (zero boundary). NumPy is used only as the
    storage format: no vectorized or library morphological operations
    are applied. The input is never mutated.

    Args:
        image: 2-D binary image with values 0 and 255.

    Returns:
        Dilated image as a new ``np.uint8`` array.
    """
    height, width = image.shape
    output = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            found_white = False

            for dy in _NEIGHBOR_OFFSETS:
                ny = y + dy
                if not 0 <= ny < height:
                    continue

                for dx in _NEIGHBOR_OFFSETS:
                    nx = x + dx
                    if not 0 <= nx < width:
                        continue

                    if image[ny, nx] == 255:
                        found_white = True
                        break

                if found_white:
                    break

            if found_white:
                output[y, x] = 255

    return output


def dilate_opencv(image: np.ndarray) -> np.ndarray:
    """Dilate a binary image using ``cv2.dilate``.

    Uses the module-level 3x3 kernel, the center anchor, a single
    iteration, and an explicit zero constant border.

    Args:
        image: 2-D binary image with values 0 and 255.

    Returns:
        Dilated image as a new ``np.uint8`` array.
    """
    return cv2.dilate(
        image,
        KERNEL_3X3,
        anchor=(1, 1),
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
