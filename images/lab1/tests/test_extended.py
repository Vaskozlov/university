"""Independent oracles, algebraic properties and file/CLI integration."""

import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from dilation_lab.dilation import dilate_opencv, dilate_python
from dilation_lab.io_utils import binarize, load_grayscale


def point_union_oracle(image):
    """Construct the union of rectangles around foreground coordinates."""
    expected = np.zeros(image.shape, dtype=np.uint8)
    height, width = image.shape
    for y, x in zip(*np.nonzero(image)):
        expected[max(0, y - 1):min(height, y + 2),
                 max(0, x - 1):min(width, x + 2)] = 255
    return expected


@pytest.mark.parametrize("dilate", [dilate_python, dilate_opencv])
def test_all_512_binary_3x3_patterns(dilate):
    for bits in range(512):
        image = np.array([(bits >> k) & 1 for k in range(9)],
                         dtype=np.uint8).reshape(3, 3) * 255
        assert np.array_equal(dilate(image), point_union_oracle(image)), bits


@pytest.mark.parametrize("dilate", [dilate_python, dilate_opencv])
@pytest.mark.parametrize("corner", [(0, 0), (0, -1), (-1, 0), (-1, -1)])
def test_every_corner(dilate, corner):
    image = np.zeros((5, 7), dtype=np.uint8)
    image[corner] = 255
    result = dilate(image)
    assert np.count_nonzero(result) == 4
    assert np.array_equal(result, point_union_oracle(image))


@pytest.mark.parametrize("dilate", [dilate_python, dilate_opencv])
def test_exactly_one_iteration(dilate):
    image = np.zeros((9, 11), dtype=np.uint8)
    image[4, 5] = 255
    result = dilate(image)
    assert np.count_nonzero(result) == 9
    assert result[4, 7] == 0
    assert np.count_nonzero(dilate(result)) == 25


@pytest.mark.parametrize("dilate", [dilate_python, dilate_opencv])
@pytest.mark.parametrize("shape", [(1, 31), (31, 1), (2, 17), (17, 2), (13, 29)])
def test_properties_on_rectangles_and_array_views(dilate, shape):
    rng = np.random.default_rng(20261030)
    for fill in (0.0, 0.01, 0.25, 0.75, 1.0):
        a = (rng.random(shape) < fill).astype(np.uint8) * 255
        b = np.maximum(a, (rng.random(shape) < 0.2).astype(np.uint8) * 255)
        # Transposed, strided and negative-stride views exercise storage layout.
        for image in (a, a.T, a[:, ::-1], a[::2, ::2]):
            before = image.copy()
            result = dilate(image)
            assert np.array_equal(result, point_union_oracle(image))
            assert np.all(result >= image)  # extensivity
            assert result.dtype == np.uint8 and result.shape == image.shape
            assert not np.shares_memory(result, image)
            assert np.array_equal(image, before)
        assert np.all(dilate(a) <= dilate(b))  # monotonicity


@pytest.mark.parametrize("dilate", [dilate_python, dilate_opencv])
def test_union_preservation(dilate):
    rng = np.random.default_rng(17)
    a = (rng.random((17, 23)) < 0.1).astype(np.uint8) * 255
    b = (rng.random((17, 23)) < 0.2).astype(np.uint8) * 255
    assert np.array_equal(dilate(np.maximum(a, b)),
                          np.maximum(dilate(a), dilate(b)))


def test_loading_color_and_threshold_endpoints(tmp_path):
    image = np.zeros((7, 11, 3), dtype=np.uint8)
    image[2:5, 3:8] = (0, 255, 255)
    path = tmp_path / "цветной.png"
    assert cv2.imwrite(str(path), image)
    gray = load_grayscale(str(path))
    assert gray.shape == image.shape[:2] and gray.dtype == np.uint8
    # PNG decoder grayscale conversion and cvtColor can round differently.
    expected = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    assert np.max(np.abs(gray.astype(np.int16) - expected.astype(np.int16))) <= 1
    assert np.array_equal(binarize(gray, 0), (gray > 0).astype(np.uint8) * 255)
    assert not np.any(binarize(gray, 255))


def test_loading_missing_and_invalid_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_grayscale(str(tmp_path / "missing.png"))
    invalid = tmp_path / "invalid.png"
    invalid.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError):
        load_grayscale(str(invalid))


def test_cli_end_to_end_and_failures(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"),
               MPLCONFIGDIR=str(tmp_path / "matplotlib"))
    image = np.zeros((13, 19), dtype=np.uint8)
    image[0, -1] = image[7, 9] = 255
    source = tmp_path / "input.png"
    assert cv2.imwrite(str(source), image)
    output = tmp_path / "results"
    command = [sys.executable, "-m", "dilation_lab", "run", "--input", str(source),
               "--output-dir", str(output)]
    success = subprocess.run(command, env=env, capture_output=True, text=True)
    assert success.returncode == 0, success.stderr
    assert "Mismatched pixels: 0" in success.stdout
    for name in ("input_binary", "dilated_python", "dilated_opencv", "difference", "comparison"):
        assert (output / f"{name}.png").is_file()
    assert np.array_equal(load_grayscale(str(output / "dilated_python.png")),
                          point_union_oracle(image))
    assert not np.any(load_grayscale(str(output / "difference.png")))
    failure = subprocess.run(command + ["--threshold", "256"], env=env,
                             capture_output=True, text=True)
    assert failure.returncode != 0 and "threshold" in failure.stderr
    source.unlink()
    missing = subprocess.run(command, env=env, capture_output=True, text=True)
    assert missing.returncode != 0 and "not found" in missing.stderr
