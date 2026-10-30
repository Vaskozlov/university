"""Command line entry point: ``python -m dilation_lab``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dilation_lab import __version__
from dilation_lab.benchmark import BATCH_SIZE, run_benchmark, write_csv
from dilation_lab.dilation import dilate_opencv, dilate_python
from dilation_lab.io_utils import binarize, load_grayscale, save_image
from dilation_lab.validation import compare_results
from dilation_lab.visualization import (
    save_comparison_image,
    save_speedup_plot,
    save_timing_plot,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with the ``run`` and ``benchmark`` subcommands."""
    parser = argparse.ArgumentParser(
        prog="dilation_lab",
        description="3x3 binary image dilation: Python vs OpenCV.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run", help="Process an image and compare both implementations"
    )
    run.add_argument("--input", required=True, help="Path to the input image")
    run.add_argument(
        "--output-dir", required=True, help="Directory for the result images"
    )
    run.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="Binarization threshold, 0..255 (default: 127)",
    )

    benchmark = subparsers.add_parser(
        "benchmark", help="Benchmark both implementations on synthetic masks"
    )
    benchmark.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[64, 128, 256, 512],
        help="Square mask side lengths in pixels (default: 64 128 256 512)",
    )
    benchmark.add_argument(
        "--runs", type=int, default=15, help="Timed runs per size (default: 15)"
    )
    benchmark.add_argument(
        "--warmups",
        type=int,
        default=3,
        help="Warmup runs per size (default: 3)",
    )
    benchmark.add_argument(
        "--output-dir",
        required=True,
        help="Directory for benchmark.csv and the plots",
    )

    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    """Run both dilations on an image, validate, and save the results."""
    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Error: input image not found: {input_path}", file=sys.stderr)
        return 1

    try:
        gray = load_grayscale(str(input_path))
        binary = binarize(gray, args.threshold)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    python_result = dilate_python(binary)
    opencv_result = dilate_opencv(binary)
    report = compare_results(python_result, opencv_result)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_image(str(output_dir / "input_binary.png"), binary)
    save_image(str(output_dir / "dilated_python.png"), python_result)
    save_image(str(output_dir / "dilated_opencv.png"), opencv_result)
    save_image(str(output_dir / "difference.png"), report["difference"])
    save_comparison_image(
        str(output_dir / "comparison.png"),
        gray,
        binary,
        python_result,
        opencv_result,
        report["difference"],
    )

    print(f"Mismatched pixels: {report['mismatched_pixels']}")
    if report["mismatched_pixels"] != 0:
        print(
            "Error: the two implementations produced different results.",
            file=sys.stderr,
        )
        return 1

    print(f"Results match. Output saved to {output_dir}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    """Benchmark both algorithms and write the CSV and plots."""
    try:
        results = run_benchmark(args.sizes, runs=args.runs, warmups=args.warmups)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark.csv"
    plot_path = output_dir / "benchmark.png"
    speedup_path = output_dir / "speedup.png"

    write_csv(results, str(csv_path))
    save_timing_plot(str(plot_path), results)
    save_speedup_plot(str(speedup_path), results)

    for r in results:
        method = (
            f"batched ({BATCH_SIZE} calls per interval)"
            if r.batched
            else "single call"
        )
        print(
            f"{r.size}x{r.size} {r.algorithm:<8} "
            f"median={r.median_ms:.6f} ms min={r.min_ms:.6f} ms "
            f"iqr={r.iqr_ms:.6f} ms runs={r.runs} method={method}"
        )
    print(f"CSV written to {csv_path}")
    print(f"Plots written to {plot_path} and {speedup_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the selected subcommand."""
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    return _cmd_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
