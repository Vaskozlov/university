"""Reproduce the extended image study and measurements used in the report.

Run from the project root: MPLCONFIGDIR=/tmp/dilation-mpl .venv/bin/python scripts/verify_lab.py
This script deliberately keeps the submitted implementations unchanged.
"""

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dilation_lab.benchmark import make_mask
from dilation_lab.dilation import dilate_opencv, dilate_python
from dilation_lab.io_utils import binarize, load_grayscale, save_image
from dilation_lab.validation import compare_results

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "verification"
FIG = OUT / "figures"
RUNS, WARMUPS, SEED = 15, 3, 42
ALGORITHMS = {"python": dilate_python, "opencv": dilate_opencv}


def fixtures():
    """Analytic, locally generated inputs with known geometric features."""
    images = [("example", "Исходный пример", load_grayscale(str(ROOT / "data/example.png")))]
    lines = np.zeros((192, 288), np.uint8)
    cv2.rectangle(lines, (25, 25), (130, 150), 255, 1)
    cv2.line(lines, (155, 30), (260, 165), 255, 1)
    cv2.line(lines, (0, 0), (287, 0), 255, 1)
    cv2.circle(lines, (210, 90), 36, 255, 1)
    lines[-1, -1] = 255
    images.append(("thin_lines", "Тонкие линии и границы", lines))
    gaps = np.zeros((144, 216), np.uint8)
    for row, gap in enumerate((1, 2, 3)):
        y = 12 + row * 44
        gaps[y:y + 28, 30:100] = 255
        gaps[y:y + 28, 100 + gap:175] = 255
    images.append(("gaps", "Зазоры 1, 2 и 3 пикселя", gaps))
    text = np.zeros((192, 384), np.uint8)
    for label, y, scale in (("DILATION", 55, 1.45), ("P3415 / LAB 1", 106, 0.86),
                            ("thin strokes 123", 155, 0.7)):
        cv2.putText(text, label, (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    255, 1, cv2.LINE_AA)
    images.append(("text", "Сглаженный текст", text))
    noise = (np.random.default_rng(SEED).random((192, 288)) < 0.01).astype(np.uint8) * 255
    images.append(("noise", "Импульсный белый шум", noise))
    y, x = np.mgrid[0:192, 0:288]
    gray = 25 + 130 * x / 287 + 85 * np.exp(-((x - 100) ** 2 + (y - 80) ** 2) / 2500)
    gray += 25 * np.sin(x / 13) * np.cos(y / 17)
    gray += np.random.default_rng(SEED).normal(0, 7, gray.shape)
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    images.append(("grayscale", "Полутоновое поле", gray))
    checker = (((x // 4 + y // 4) % 2) * 255).astype(np.uint8)
    images.append(("checkerboard", "Шахматная сетка", checker))
    return images


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def image_study():
    rows, panels = [], []
    for name, title, gray in fixtures():
        destination = OUT / "images" / name
        destination.mkdir(parents=True, exist_ok=True)
        if name != "example":
            generated = ROOT / "data" / "test_images"
            generated.mkdir(parents=True, exist_ok=True)
            save_image(str(generated / f"{name}.png"), gray)
        binary = binarize(gray, 127)
        before = binary.copy()
        python, opencv = dilate_python(binary), dilate_opencv(binary)
        comparison = compare_results(python, opencv)
        assert comparison["identical"] and comparison["dtype_match"]
        assert np.array_equal(binary, before) and np.all(python >= binary)
        for suffix, image in (("original", gray), ("binary", binary), ("python", python),
                              ("opencv", opencv), ("difference", comparison["difference"])):
            save_image(str(destination / f"{suffix}.png"), image)
        rows.append(dict(name=name, height=gray.shape[0], width=gray.shape[1],
                         pixels=gray.size, white_before=int(np.count_nonzero(binary)),
                         white_after=int(np.count_nonzero(python)),
                         components_before=cv2.connectedComponents(binary, connectivity=8)[0] - 1,
                         components_after=cv2.connectedComponents(python, connectivity=8)[0] - 1,
                         mismatched_pixels=comparison["mismatched_pixels"],
                         max_abs_difference=comparison["max_abs_difference"],
                         input_sha256=hashlib.sha256(gray.tobytes()).hexdigest()))
        panels.append((title, gray, binary, python, comparison["difference"]))
    write_csv(OUT / "image_metrics.csv", rows)
    for filename, group in (("gallery_geometry", panels[:4]), ("gallery_texture", panels[4:])):
        fig, axes = plt.subplots(len(group), 4, figsize=(10, len(group) * 2.1), squeeze=False)
        for index, (title, *images) in enumerate(group):
            for col, image in enumerate(images):
                axes[index, col].imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                axes[index, col].set_xticks([])
                axes[index, col].set_yticks([])
                if index == 0:
                    axes[index, col].set_title(("Полутоновый вход", "Бинаризация", "Дилатация", "Разность")[col])
            axes[index, 0].set_ylabel(title, fontsize=10)
        fig.tight_layout(pad=0.8)
        fig.savefig(FIG / f"{filename}.png", dpi=220)
        plt.close(fig)
    # Magnified lattice: each cell corresponds to exactly one original pixel.
    a = np.zeros((9, 13), np.uint8)
    a[4, 3] = a[4, 7] = a[0, 12] = 255
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, image, title in zip(axes, (a, dilate_python(a)), ("Вход", "Один проход, квадрат 3×3")):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_xticks(np.arange(-0.5, 13, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 9, 1), minor=True)
        ax.grid(which="minor", color="#8b8b8b", linewidth=0.5)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG / "pixel_example.png", dpi=220)
    plt.close(fig)
    # Expose gap closure at the exact center of each pair of rectangles.
    gap_input = panels[2][2]
    gap_output = dilate_python(gap_input)
    gaps = [dict(gap=gap, remaining_black=int(np.count_nonzero(gap_output[26 + row * 44, 100:100 + gap] == 0)))
            for row, gap in enumerate((1, 2, 3))]
    thresholds = []
    gray = fixtures()[5][2]
    fig, axes = plt.subplots(2, 3, figsize=(9, 4.6))
    for col, threshold in enumerate((80, 127, 180)):
        binary = binarize(gray, threshold)
        result = dilate_python(binary)
        assert np.array_equal(result, dilate_opencv(binary))
        thresholds.append(dict(threshold=threshold, white_before=int(np.count_nonzero(binary)),
                               white_after=int(np.count_nonzero(result)), pixels=gray.size))
        for row, image in enumerate((binary, result)):
            axes[row, col].imshow(image, cmap="gray", vmin=0, vmax=255)
            axes[row, col].axis("off")
        axes[0, col].set_title(f"Порог {threshold}")
    fig.tight_layout()
    fig.savefig(FIG / "thresholds.png", dpi=220)
    plt.close(fig)
    write_csv(OUT / "threshold_metrics.csv", thresholds)
    return rows, gaps, thresholds


def time_call(function, image, batch):
    start = time.perf_counter_ns()
    for _ in range(batch):
        result = function(image)
    return (time.perf_counter_ns() - start) / 1e6 / batch, result


def measure(image, series, raw):
    assert np.array_equal(dilate_python(image), dilate_opencv(image))
    rows = []
    for name, function in ALGORITHMS.items():
        calibration = [time_call(function, image, 1)[0] for _ in range(WARMUPS)]
        batch = min(4096, max(1, math.ceil(20 / max(statistics.median(calibration), 1e-6))))
        # Warm the selected batch once; all calibration calls are excluded.
        time_call(function, image, batch)
        samples = []
        for run in range(RUNS):
            elapsed, result = time_call(function, image, batch)
            samples.append(elapsed)
            raw.append(dict(series=series, algorithm=name, run=run + 1,
                            calls_per_interval=batch, ms_per_call=elapsed))
        q1, _, q3 = statistics.quantiles(samples, n=4, method="inclusive")
        row = dict(series=series, algorithm=name, height=image.shape[0], width=image.shape[1],
                   pixels=image.size, fill_actual=float(np.count_nonzero(image) / image.size),
                   median_ms=statistics.median(samples), min_ms=min(samples),
                   q1_ms=q1, q3_ms=q3, iqr_ms=q3-q1, runs=RUNS, calls_per_interval=batch)
        rows.append(row)
        print(f"{series:15} {name:6} median={row['median_ms']:.6f} ms batch={batch}", flush=True)
    return rows


def timing_study():
    raw, size_rows, density_rows = [], [], []
    for size in (64, 128, 256, 512, 1024):
        size_rows.extend(measure(make_mask(size, SEED, 0.25), f"size_{size}", raw))
    for fill in (0, 0.01, 0.1, 0.25, 0.5, 1):
        density_rows.extend(measure(make_mask(256, SEED, fill), f"fill_{fill:g}", raw))
    write_csv(OUT / "benchmark_sizes.csv", size_rows)
    write_csv(OUT / "benchmark_density.csv", density_rows)
    write_csv(OUT / "timing_samples.csv", raw)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, marker, label in (("python", "o", "Python: циклы"), ("opencv", "s", "OpenCV: 1 поток")):
        items = [r for r in size_rows if r["algorithm"] == name]
        medians = np.array([r["median_ms"] for r in items])
        errors = [medians - [r["q1_ms"] for r in items], [r["q3_ms"] for r in items] - medians]
        axes[0].errorbar([r["pixels"] for r in items], medians, yerr=errors,
                         fmt=marker + "-", capsize=3, label=label)
    speedups = [size_rows[k]["median_ms"] / size_rows[k+1]["median_ms"] for k in range(0, len(size_rows), 2)]
    axes[1].plot((64, 128, 256, 512, 1024), speedups, "o-", color="#1e6b45")
    axes[0].set(xscale="log", yscale="log", xlabel="Количество пикселей", ylabel="Медиана, мс",
                title="Время и межквартильный размах")
    axes[1].set(xscale="log", xlabel="Сторона изображения, пикс.", ylabel="T(Python) / T(OpenCV)",
                title="Ускорение OpenCV")
    axes[0].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "performance.png", dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for ax, name, multiplier, unit in zip(axes, ALGORITHMS, (1, 1000), ("мс", "мкс")):
        items = [r for r in density_rows if r["algorithm"] == name]
        ax.plot([r["fill_actual"] * 100 for r in items], [r["median_ms"] * multiplier for r in items], "o-")
        ax.set(xlabel="Белые пиксели на входе, %", ylabel=f"Медиана, {unit}",
               title=f"{name}: 256×256")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "density.png", dpi=220)
    plt.close(fig)
    return size_rows, density_rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    # GCD on macOS accepts only non-positive settings. Zero disables
    # OpenCV parallel regions; verify the effective value before timing.
    cv2.setNumThreads(0)
    assert cv2.getNumThreads() == 1, "Could not disable OpenCV parallel regions"
    cv2.ocl.setUseOpenCL(False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    test_run = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT,
                              capture_output=True, text=True, timeout=120)
    (OUT / "pytest.txt").write_text(test_run.stdout + test_run.stderr, encoding="utf-8")
    if test_run.returncode:
        raise RuntimeError(f"Tests failed; see {OUT / 'pytest.txt'}")
    print(test_run.stdout.strip(), flush=True)
    image_rows, gaps, thresholds = image_study()
    size_rows, density_rows = timing_study()
    metadata = dict(timestamp=datetime.now(ZoneInfo("Europe/Moscow")).isoformat(),
                    python=sys.version, platform=platform.platform(), machine=platform.machine(),
                    processor=platform.processor(), opencv_threads=cv2.getNumThreads(),
                    opencl=cv2.ocl.useOpenCL(), opencv_optimized=cv2.useOptimized(),
                    seed=SEED, runs=RUNS, warmups=WARMUPS, target_interval_ms=20,
                    opencv_threads_setting=0,
                    pytest_summary=test_run.stdout.strip().splitlines()[-1],
                    versions={name: importlib.metadata.version(name) for name in
                              ("numpy", "opencv-python-headless", "matplotlib", "pytest")})
    (OUT / "opencv_build.txt").write_text(cv2.getBuildInformation(), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(dict(environment=metadata, images=image_rows,
                                                gaps=gaps, thresholds=thresholds,
                                                sizes=size_rows, density=density_rows),
                                               ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(environment=metadata, gaps=gaps), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
