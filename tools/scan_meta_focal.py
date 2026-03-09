#!/usr/bin/env python3
"""Recursively scan meta.json files and extract focal intrinsics fx/fy.

Each first-level folder under the root is treated as one dataset.
For each dataset, duplicate (fx, fy) pairs are collapsed in summary output.
The script can also aggregate fx image counts and plot distributions.

meta.json example:
  {"intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]}

Usage:
  python scan_meta_focal.py /path/to/eval   # e.g. MoGe/data/eval/
  python scan_meta_focal.py /path/to/root --output focals.csv  # CSV columns: dataset,fx,fy
  python scan_meta_focal.py /path/to/root -q   # summary only
  python scan_meta_focal.py /path/to/root --plot   # aggregate and plot fx distribution
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


def get_visualization_dir() -> Path:
    """Return the standard output directory for generated plots/statistics."""
    out_dir = Path(os.getcwd()) / "artifacts" / "visualization"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def find_meta_jsons(root: str):
    """Recursively find all meta.json under root."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    return sorted(root.rglob("meta.json"))


def get_fx_fy(meta_path: str) -> tuple[float, float] | None:
    """Read fx/fy from meta.json (intrinsics[0][0], intrinsics[1][1]); None if invalid."""
    try:
        with open(meta_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    intrinsics = data.get("intrinsics")
    if not intrinsics or len(intrinsics) < 2:
        return None
    try:
        fx = float(intrinsics[0][0])
        fy = float(intrinsics[1][1])
        return (fx, fy)
    except (IndexError, TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Recursively scan meta.json files and extract fx/fy"
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=os.getcwd(),
        help="Root directory to scan (default: current directory)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Optional: write results to CSV (dataset,fx,fy), unique per dataset/focal pair",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Summary/problem output only (no per-file lines)",
    )
    parser.add_argument(
        "--plot", "-p",
        action="store_true",
        help="Count images for each fx and plot distribution",
    )
    parser.add_argument(
        "--plot-output",
        default=None,
        help="Output path for distribution plot (default: artifacts/visualization/fx_distribution.png)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: 50)",
    )
    args = parser.parse_args()

    meta_files = find_meta_jsons(args.root)
    if not meta_files:
        print(f"No meta.json files found under: {args.root}")
        return

    results = []  # (dataset_name, rel, fx, fy)
    failed = []
    root_resolved = Path(args.root).resolve()

    for path in meta_files:
        try:
            rel = path.relative_to(root_resolved)
            rel_str = str(rel)
            # Use first-level folder name as dataset name.
            dataset_name = rel.parts[0] if rel.parts else root_resolved.name
        except ValueError:
            rel_str = os.path.relpath(path, args.root)
            dataset_name = root_resolved.name or str(root_resolved)
        pair = get_fx_fy(str(path))
        if pair is None:
            failed.append(rel_str)
            continue
        fx, fy = pair
        results.append((dataset_name, rel_str, fx, fy))
        if not args.quiet:
            print(f"{rel_str}\tfx={fx:.3f}\tfy={fy:.3f}")

    if failed and not args.quiet:
        print("\nFiles with unreadable fx/fy:")
        for r in failed:
            print(f"  {r}")

    # Dataset-level summary: unique (fx, fy) entries per dataset.
    unique_by_dataset_focal = list(dict.fromkeys((r[0], r[2], r[3]) for r in results))
    unique_by_dataset_focal.sort(key=lambda x: (x[0], x[1], x[2]))
    print(f"\nTotal meta.json: {len(meta_files)}, succeeded: {len(results)}, failed: {len(failed)}")
    print("\nDataset\tfx\tfy")
    for dataset_name, fx, fy in unique_by_dataset_focal:
        print(f"{dataset_name}\t{fx:.3f}\t{fy:.3f}")

    vis_dir = get_visualization_dir()

    if args.output and results:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("dataset\tfx\tfy\n")
            for dataset_name, fx, fy in unique_by_dataset_focal:
                f.write(f"{dataset_name}\t{fx:.3f}\t{fy:.3f}\n")
        print(f"\nWritten: {out_path}")

    # fx count summary by (dataset, fx, fy), saved to CSV.
    if results:
        # (dataset, fx, fy) -> count, with fx/fy rounded to 3 decimals.
        focal_counts = Counter((r[0], round(r[2], 3), round(r[3], 3)) for r in results)
        print("\n--- fx image counts by dataset ---")
        print("Dataset\tfx\tfy\tImage count")
        stats_path = vis_dir / "fx_stats.csv"
        with open(stats_path, "w") as f:
            f.write("dataset,fx,fy,count\n")
            for (dataset_name, fx, fy) in sorted(focal_counts.keys()):
                c = focal_counts[(dataset_name, fx, fy)]
                f.write(f"{dataset_name},{fx:.3f},{fy:.3f},{c}\n")
                print(f"{dataset_name}\t{fx:.3f}\t{fy:.3f}\t{c}")
        print(f"\nSaved fx stats: {stats_path}")

    # Plot fx distribution: one curve per dataset in a single figure.
    if args.plot and results:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("\nmatplotlib is not installed. Install with: pip install matplotlib")
        else:
            # Group fx values by dataset.
            dataset_fx = defaultdict(list)
            for dataset_name, _rel, fx, _fy in results:
                dataset_fx[dataset_name].append(round(fx, 3))

            # Shared bins based on global fx range.
            all_fx = [r[2] for r in results]
            fx_min, fx_max = min(all_fx), max(all_fx)
            bins = np.linspace(fx_min, fx_max, args.bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.tab10(np.linspace(0, 1, max(len(dataset_fx), 1)))
            for idx, (dataset_name, fx_list) in enumerate(sorted(dataset_fx.items())):
                hist, _ = np.histogram(fx_list, bins=bins)
                # Keep non-empty bins only; y-axis uses ln(count).
                mask = hist > 0
                x_plot = bin_centers[mask]
                y_plot = np.log(hist[mask].astype(float))
                if len(x_plot) > 0:
                    ax.plot(x_plot, y_plot, "o-", label=dataset_name, color=colors[idx % 10], alpha=0.8, markersize=4)
            ax.set_xlabel("fx")
            ax.set_ylabel("ln(Image count)")
            ax.set_title("fx distribution by dataset")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = Path(args.plot_output) if args.plot_output else vis_dir / "fx_distribution.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Saved fx distribution plot: {plot_path}")


if __name__ == "__main__":
    main()
