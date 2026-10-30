"""Image loading, binarization, and saving."""

from __future__ import annotations

import os

import cv2
import numpy as np


def load_grayscale(path: str) -> np.ndarray:
    """Load an image file as a 2-D grayscale array.

    Args:
        path: Path to the image file (color or grayscale).

    Returns:
        Grayscale image as a 2-D ``np.uint8`` array.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file exists but cannot be decoded as an image.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not decode image file: {path}")
    return image


def binarize(gray: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Binarize a grayscale image with a hard threshold.

    A pixel becomes white (255) only if its value is strictly greater
    than ``threshold``; all other pixels become 0.

    Args:
        gray: 2-D grayscale image.
        threshold: Threshold value, must be in the range 0..255.

    Returns:
        New binary image with values 0 and 255.

    Raises:
        ValueError: If ``threshold`` is outside 0..255.
    """
    if not 0 <= threshold <= 255:
        raise ValueError(f"threshold must be in the range 0..255, got {threshold}")
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def save_image(path: str, image: np.ndarray) -> None:
    """Save an image array to a file (format inferred from the extension).

    Args:
        path: Destination file path.
        image: Image array to save.

    Raises:
        ValueError: If the file could not be written.
    """
    if not cv2.imwrite(path, image):
        raise ValueError(f"Could not save image to: {path}")
