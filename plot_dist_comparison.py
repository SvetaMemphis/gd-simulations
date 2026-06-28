"""
Plot a dist-comparison CSV produced by main_disks.py.

Usage
-----
  python plot_dist_comparison.py <csv_file> [options]

Examples
--------
  python plot_dist_comparison.py experiment_disks_dist_comparison_adam_k10_lr0.1_n20_d5_runs3.csv
  python plot_dist_comparison.py results.csv --output my_plot.png --ymin 0.8 --ymax 1.0
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parse_tag_from_filename(stem: str) -> str:
    """Extract the trailing tag (optimizer/k/lr/n/d/runs) from the filename stem."""
    m = re.search(r"_((?:gd|adam)_.+)$", stem, re.IGNORECASE)
    return m.group(1).replace("_", "  ") if m else stem


def plot_dist_comparison(
    csv_path: str,
    output_png: str = "",
    ymin: float = 0.5,
    ymax: float = 1.0,
    title: str = "",
) -> None:
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "run":            int(row["run"]),
                "step":           int(row["step"]),
                "min_dist_large": float(row["min_dist_large"]),
                "min_dist_small": float(row["min_dist_small"]),
            })

    if not rows:
        print("CSV is empty — nothing to plot.")
        return

    num_runs = len({r["run"] for r in rows})

    stem = Path(csv_path).stem
    if not output_png:
        output_png = str(Path(csv_path).with_suffix(".png"))
    if not title:
        title = "Min boundary distance vs steps  —  " + _parse_tag_from_filename(stem)

    fig, ax = plt.subplots(figsize=(10, 6))

    if num_runs <= 5:
        linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
        by_run: dict = defaultdict(lambda: {"steps": [], "large": [], "small": []})
        for row in rows:
            by_run[row["run"]]["steps"].append(row["step"])
            by_run[row["run"]]["large"].append(row["min_dist_large"])
            by_run[row["run"]]["small"].append(row["min_dist_small"])

        for i, run_id in enumerate(sorted(by_run)):
            ls = linestyles[i % len(linestyles)]
            run_label = f" (run {run_id})" if num_runs > 1 else ""
            ax.plot(by_run[run_id]["steps"], by_run[run_id]["large"],
                    color="steelblue", linestyle=ls,
                    label=f"Large network{run_label}")
            ax.plot(by_run[run_id]["steps"], by_run[run_id]["small"],
                    color="tomato", linestyle=ls,
                    label=f"Small network (1D, k=2){run_label}")
    else:
        by_step: dict = defaultdict(lambda: {"large": [], "small": []})
        for row in rows:
            by_step[row["step"]]["large"].append(row["min_dist_large"])
            by_step[row["step"]]["small"].append(row["min_dist_small"])

        steps_sorted = sorted(by_step)
        mean_large = np.array([float(np.nanmean(by_step[s]["large"])) for s in steps_sorted])
        mean_small = np.array([float(np.nanmean(by_step[s]["small"])) for s in steps_sorted])
        std_large  = np.array([float(np.nanstd(by_step[s]["large"]))  for s in steps_sorted])
        std_small  = np.array([float(np.nanstd(by_step[s]["small"]))  for s in steps_sorted])

        ax.plot(steps_sorted, mean_large, color="steelblue", label="Large network")
        ax.plot(steps_sorted, mean_small, color="tomato",    label="Small network (1D, k=2)")
        ax.fill_between(steps_sorted, mean_large - std_large, mean_large + std_large,
                        color="steelblue", alpha=0.2)
        ax.fill_between(steps_sorted, mean_small - std_small, mean_small + std_small,
                        color="tomato", alpha=0.2)

    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Step (Phase 2)")
    ax.set_ylabel("Min boundary distance")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_file", help="Path to the dist-comparison CSV")
    parser.add_argument("--output", "-o", default="",
                        help="Output PNG path (default: same name as CSV with .png)")
    parser.add_argument("--ymin", type=float, default=0.5, help="Y-axis lower bound (default: 0.5)")
    parser.add_argument("--ymax", type=float, default=1.0, help="Y-axis upper bound (default: 1.0)")
    parser.add_argument("--title", default="", help="Custom plot title")
    args = parser.parse_args()

    plot_dist_comparison(
        csv_path=args.csv_file,
        output_png=args.output,
        ymin=args.ymin,
        ymax=args.ymax,
        title=args.title,
    )


if __name__ == "__main__":
    main()
