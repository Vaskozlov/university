"""Timing of the two dilation implementations and CSV export."""

from __future__ import annotations

import csv
import statistics
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from dilation_lab.dilation import dilate_opencv, dilate_python

#: Number of calls timed in one interval when a single call is too fast.
BATCH_SIZE = 5
#: A warmup median below this (ms) switches a size to batched timing.
BATCH_THRESHOLD_MS = 2.0

#: Exact column order of benchmark.csv.
CSV_COLUMNS = (
    "size",
    "width",
    "height",
    "pixels",
    "algorithm",
    "median_ms",
    "min_ms",
    "iqr_ms",
    "runs",
)

_ALGORITHMS: tuple[tuple[str, Callable[[np.ndarray], np.ndarray]], ...] = (
    ("python", dilate_python),
    ("opencv", dilate_opencv),
)


@dataclass(frozen=True)
class BenchmarkResult:
    """Timing statistics for one algorithm at one image size."""

    size: int
    width: int
    height: int
    pixels: int
    algorithm: str
    median_ms: float
    min_ms: float
    iqr_ms: float
    runs: int
    batched: bool


def make_mask(size: int, seed: int = 42, fill: float = 0.25) -> np.ndarray:
    """Create a reproducible synthetic binary mask.

    Args:
        size: Side length of the square mask in pixels.
        seed: Seed for ``np.random.default_rng``.
        fill: Fraction of white pixels (about 25% by default).

    Returns:
        ``size x size`` ``np.uint8`` array with values 0 and 255.
    """
    rng = np.random.default_rng(seed)
    white = rng.random((size, size)) < fill
    return (white * 255).astype(np.uint8)


def _time_single(
    dilation: Callable[[np.ndarray], np.ndarray], image: np.ndarray
) -> tuple[float, np.ndarray]:
    """Time one dilation call; returns elapsed milliseconds and the output."""
    start = time.perf_counter_ns()
    output = dilation(image)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
    return elapsed_ms, output


def _time_batch(
    dilation: Callable[[np.ndarray], np.ndarray],
    image: np.ndarray,
    batch_size: int,
) -> tuple[float, np.ndarray]:
    """Time ``batch_size`` calls in one interval; returns the per-call mean."""
    start = time.perf_counter_ns()
    output = None
    for _ in range(batch_size):
        output = dilation(image)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000 / batch_size
    return elapsed_ms, output


def _iqr_ms(samples: list[float]) -> float:
    """Interquartile range in milliseconds (0.0 if fewer than 2 samples)."""
    if len(samples) < 2:
        return 0.0
    quartiles = statistics.quantiles(samples, n=4)
    return quartiles[2] - quartiles[0]


def run_benchmark(
    sizes: list[int], runs: int = 15, warmups: int = 3
) -> list[BenchmarkResult]:
    """Benchmark both dilation algorithms on synthetic masks.

    For each size a mask is created before any timing starts. Both
    algorithms get the same number of warmups (excluded from the
    statistics) and timed runs. If the warmup median of a single call
    is under ``BATCH_THRESHOLD_MS``, that size is timed in batches of
    ``BATCH_SIZE`` calls per interval for BOTH algorithms, so the
    comparison stays fair.

    Args:
        sizes: Side lengths of the square masks in pixels.
        runs: Number of timed runs per size and algorithm.
        warmups: Number of untimed warmup runs per size and algorithm.

    Returns:
        One ``BenchmarkResult`` per (size, algorithm), in input order.

    Raises:
        ValueError: If any size is less than 1, or ``runs``/``warmups`` < 1.
    """
    if any(size < 1 for size in sizes):
        raise ValueError(f"all sizes must be >= 1, got {sizes}")
    if runs < 1 or warmups < 1:
        raise ValueError(f"runs and warmups must be >= 1, got runs={runs}, warmups={warmups}")

    results: list[BenchmarkResult] = []
    for size in sizes:
        image = make_mask(size)

        # Warm up both algorithms with single timed calls.
        warmup_medians: dict[str, float] = {}
        for name, dilation in _ALGORITHMS:
            samples = [
                _time_single(dilation, image)[0] for _ in range(warmups)
            ]
            warmup_medians[name] = statistics.median(samples)

        batched = any(m < BATCH_THRESHOLD_MS for m in warmup_medians.values())

        for name, dilation in _ALGORITHMS:
            samples: list[float] = []
            output: np.ndarray | None = None
            for _ in range(runs):
                if batched:
                    elapsed_ms, output = _time_batch(dilation, image, BATCH_SIZE)
                else:
                    elapsed_ms, output = _time_single(dilation, image)
                samples.append(elapsed_ms)

            results.append(
                BenchmarkResult(
                    size=size,
                    width=size,
                    height=size,
                    pixels=size * size,
                    algorithm=name,
                    median_ms=statistics.median(samples),
                    min_ms=min(samples),
                    iqr_ms=_iqr_ms(samples),
                    runs=runs,
                    batched=batched,
                )
            )
    return results


def write_csv(results: list[BenchmarkResult], path: str) -> None:
    """Write benchmark results to a CSV file with the fixed column layout.

    Args:
        results: Results from :func:`run_benchmark`.
        path: Destination file path.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for r in results:
            writer.writerow(
                [
                    r.size,
                    r.width,
                    r.height,
                    r.pixels,
                    r.algorithm,
                    f"{r.median_ms:.6f}",
                    f"{r.min_ms:.6f}",
                    f"{r.iqr_ms:.6f}",
                    r.runs,
                ]
            )
