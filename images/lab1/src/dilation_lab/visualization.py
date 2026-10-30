"""Result images and timing plots (headless: Agg backend)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from dilation_lab.benchmark import BenchmarkResult


def save_comparison_image(
    path: str,
    original: np.ndarray,
    binary: np.ndarray,
    python_result: np.ndarray,
    opencv_result: np.ndarray,
    difference: np.ndarray,
) -> None:
    """Save a five-panel comparison image.

    Panels: original grayscale, binary input, Python result, OpenCV
    result, and the difference mask (255 where the results differ).

    Args:
        path: Destination PNG path.
        original: Original grayscale image.
        binary: Binarized input.
        python_result: Result of the Python implementation.
        opencv_result: Result of the OpenCV implementation.
        difference: Difference mask from ``compare_results``.
    """
    panels = (
        (original, "Original (grayscale)"),
        (binary, "Binary input"),
        (python_result, "Dilation: Python"),
        (opencv_result, "Dilation: OpenCV"),
        (difference, "Difference (255 = mismatch)"),
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.5))
    for ax, (image, title) in zip(axes, panels):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_timing_plot(path: str, results: list[BenchmarkResult]) -> None:
    """Save a timing plot: median time (ms) vs pixel count.

    Both algorithms are drawn on the same axes with explicitly labeled
    units. A log scale is used when the measured times span more than
    two orders of magnitude.

    Args:
        path: Destination PNG path.
        results: Results from ``run_benchmark``.
    """
    python = [r for r in results if r.algorithm == "python"]
    opencv = [r for r in results if r.algorithm == "opencv"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [r.pixels for r in python],
        [r.median_ms for r in python],
        "o-",
        label="Python (explicit loops)",
    )
    ax.plot(
        [r.pixels for r in opencv],
        [r.median_ms for r in opencv],
        "s-",
        label="OpenCV (cv2.dilate)",
    )
    ax.set_xlabel("Image size (pixels)")
    ax.set_ylabel("Median time (ms)")
    ax.set_title("Dilation 3x3: median runtime vs image size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    positive = [r.median_ms for r in results if r.median_ms > 0]
    if positive and max(positive) / min(positive) > 100:
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_speedup_plot(path: str, results: list[BenchmarkResult]) -> None:
    """Save a speedup plot: Python time / OpenCV time per image size.

    Args:
        path: Destination PNG path.
        results: Results from ``run_benchmark``.
    """
    by_size: dict[int, dict[str, float]] = {}
    for r in results:
        by_size.setdefault(r.size, {})[r.algorithm] = r.median_ms

    sizes = sorted(s for s in by_size if by_size[s].get("opencv", 0) > 0)
    speedups = [by_size[s]["python"] / by_size[s]["opencv"] for s in sizes]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([str(s) for s in sizes], speedups, color="steelblue")
    ax.set_xlabel("Image size (side, pixels)")
    ax.set_ylabel("Speedup (Python time / OpenCV time)")
    ax.set_title("OpenCV speedup over the Python implementation")
    for i, value in enumerate(speedups):
        ax.text(i, value, f"{value:.0f}x", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
