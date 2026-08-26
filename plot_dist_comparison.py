"""
Plot a dist-comparison CSV produced by main_disks.py.

If num-runs (or the number of runs found in the CSV) is <= 5, each run is
plotted individually. Otherwise, the runs are aggregated into a mean line
with a shaded +/- std band.

Usage
-----
  python plot_dist_comparison.py <csv_file> [options]

Examples
--------
  python plot_dist_comparison.py experiment_disks_dist_comparison_adam_k10_lr0.1_n20_d5_runs3.csv
  python plot_dist_comparison.py results.csv --num-runs 10 --output my_plot.png --ymin 0.8 --ymax 1.0
  python plot_dist_comparison.py results.csv --save-tex
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplot2tikz import save as save_tikz


def _parse_tag_from_filename(stem: str) -> str:
    """Extract the trailing tag (optimizer/k/lr/n/d/runs) from the filename stem."""
    m = re.search(r"_((?:gd|adam)_.+)$", stem, re.IGNORECASE)
    return m.group(1).replace("_", "  ") if m else stem


_ASCII_REPLACEMENTS = {
    "—": "-",   # em dash
    "–": "-",   # en dash
    "‘": "'",   # left single quote
    "’": "'",   # right single quote
    "“": '"',   # left double quote
    "”": '"',   # right double quote
    "…": "...", # ellipsis
    "≤": "<=",  # less-than-or-equal
    "≥": ">=",  # greater-than-or-equal
    "±": "+/-", # plus-minus
}


def _clean_tex(tex_path: str) -> None:
    """Rewrite a .tex file in place: drop non-ASCII characters and blank lines."""
    raw = Path(tex_path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    for ch, replacement in _ASCII_REPLACEMENTS.items():
        text = text.replace(ch, replacement)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    lines = [line for line in text.splitlines() if line.strip()]
    Path(tex_path).write_text("\n".join(lines) + "\n", encoding="ascii")


def plot_dist_comparison(
    csv_path: str,
    num_runs: int = 0,
    output_png: str = "",
    ymin: float = 0.5,
    ymax: float = 1.0,
    title: str = "",
    save_tex: bool = False,
    output_tex: str = "",
) -> None:
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # min_dist_small is blank for Phase-1-only rows in a unified-comparison CSV,
            # since the small network doesn't exist yet; treat it as a gap (NaN).
            small_str = row["min_dist_small"].strip()
            rows.append({
                "run":            int(row["run"]),
                "step":           int(row["step"]),
                "min_dist_large": float(row["min_dist_large"]),
                "min_dist_small": float(small_str) if small_str else float("nan"),
            })

    if not rows:
        print("CSV is empty — nothing to plot.")
        return

    available_runs = sorted({r["run"] for r in rows})
    selected_runs = available_runs if num_runs <= 0 else available_runs[:num_runs]
    rows = [r for r in rows if r["run"] in selected_runs]
    num_runs = len(selected_runs)

    stem = Path(csv_path).stem
    if not output_png:
        output_png = str(Path(csv_path).with_name(f"{stem}_runs{num_runs}.png"))
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
        half_std_large = 0.5 * std_large
        half_std_small = 0.5 * std_small
        ax.fill_between(steps_sorted, mean_large - half_std_large, mean_large + half_std_large,
                        color="steelblue", alpha=0.2)
        ax.fill_between(steps_sorted, mean_small - half_std_small, mean_small + half_std_small,
                        color="tomato", alpha=0.2)

    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Step (Phase 2)")
    ax.set_ylabel("Min boundary distance")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    print(f"Plotted {num_runs}/{len(available_runs)} runs.")
    print(f"Saved: {output_png}")

    if save_tex or output_tex:
        if not output_tex:
            output_tex = str(Path(csv_path).with_name(f"{stem}_runs{num_runs}.tex"))
        save_tikz(output_tex, figure=fig)
        _clean_tex(output_tex)
        print(f"Saved: {output_tex}")

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_file", help="Path to the dist-comparison CSV")
    parser.add_argument("--num-runs", type=int, default=0,
                        help="Number of runs to take from the CSV, in run-id order "
                             "(default: use all runs in the file)")
    parser.add_argument("--output", "-o", default="",
                        help="Output PNG path (default: <csv name>_runs<N>.png)")
    parser.add_argument("--ymin", type=float, default=0.5, help="Y-axis lower bound (default: 0.5)")
    parser.add_argument("--ymax", type=float, default=1.0, help="Y-axis upper bound (default: 1.0)")
    parser.add_argument("--title", default="", help="Custom plot title")
    parser.add_argument("--save-tex", nargs="?", const="", default=None, metavar="PATH",
                        help="Also save a .tex (TikZ) version via matplot2tikz "
                             "(default path: <csv name>_runs<N>.tex)")
    args = parser.parse_args()

    plot_dist_comparison(
        csv_path=args.csv_file,
        num_runs=args.num_runs,
        output_png=args.output,
        ymin=args.ymin,
        ymax=args.ymax,
        title=args.title,
        save_tex=args.save_tex is not None,
        output_tex=args.save_tex or "",
    )


if __name__ == "__main__":
    main()
