#!/usr/bin/env python3
"""Recursively scan all depth.png files and summarize depth distribution.

Each first-level folder under the root is treated as one dataset.
The script can also aggregate image counts by depth bins and plot distributions.

depth.png is assumed to use the same 16-bit encoding decoded by read_depth.py.

Usage:
  python scan_depth_distribution.py /path/to/root   # e.g. MoGe/data/eval/ or processed/
  python scan_depth_distribution.py /path/to/root --output depths.csv
  python scan_depth_distribution.py /path/to/root -q   # summary only
  python scan_depth_distribution.py /path/to/root --plot   # plot depth distribution
"""

import argparse
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

# Allow importing sibling read_depth.py from any working directory.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

try:
    from read_depth import read_depth_png
except ImportError:
    read_depth_png = None

import numpy as np


def get_visualization_dir() -> Path:
    """Return the standard output directory for generated plots/statistics."""
    out_dir = Path(os.getcwd()) / "artifacts" / "visualization"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def find_depth_pngs(root: str):
    """Recursively find all depth.png under root."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    return sorted(root.rglob("depth.png"))


def get_depth_stats(depth_path: str) -> dict | None:
    """Decode one depth.png and return stats for that frame; None on failure."""
    if read_depth_png is None:
        raise RuntimeError("Failed to import read_depth.read_depth_png. Ensure read_depth.py is in the same directory.")
    try:
        depth_meters, near, far = read_depth_png(depth_path)
    except Exception:
        return None
    valid_mask = np.isfinite(depth_meters)
    valid_count = int(np.sum(valid_mask))
    total = depth_meters.size
    if valid_count == 0:
        return {
            "depth_min": None,
            "depth_max": None,
            "depth_mean": None,
            "valid_ratio": 0.0,
            "near": near,
            "far": far,
        }
    valid_depths = depth_meters[valid_mask]
    return {
        "depth_min": float(np.min(valid_depths)),
        "depth_max": float(np.max(valid_depths)),
        "depth_mean": float(np.mean(valid_depths)),
        "valid_ratio": valid_count / total,
        "near": near,
        "far": far,
    }


def _worker(args: tuple) -> tuple:
    """Multiprocessing worker for one depth.png.

    Returns:
        (relative_path, dataset_name, stats_dict_or_none)
    """
    depth_path_str, root_str = args
    root_resolved = Path(root_str).resolve()
    path = Path(depth_path_str)
    try:
        rel = path.relative_to(root_resolved)
        rel_str = str(rel)
        dataset_name = rel.parts[0] if rel.parts else root_resolved.name
    except ValueError:
        rel_str = os.path.relpath(path, root_str)
        dataset_name = root_resolved.name or str(root_resolved)
    stats = get_depth_stats(depth_path_str)
    if stats is None or stats.get("depth_min") is None:
        return (rel_str, dataset_name, None)
    return (rel_str, dataset_name, stats)


def main():
    parser = argparse.ArgumentParser(
        description="Recursively scan depth.png files and summarize depth distribution"
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
        help="Optional: write per-frame results to CSV (dataset,rel,depth_min,depth_max,depth_mean,valid_ratio)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Summary/problem output only (no per-file lines)",
    )
    parser.add_argument(
        "--plot", "-p",
        action="store_true",
        help="Aggregate depth values and generate a distribution plot",
    )
    parser.add_argument(
        "--plot-output",
        default=None,
        help="Output path for distribution plot (default: artifacts/visualization/depth_distribution.png)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: 50)",
    )
    parser.add_argument(
        "--depth-max",
        type=float,
        default=None,
        help="Maximum x-axis depth (meters); values beyond are clipped to the last bin",
    )
    parser.add_argument(
        "--workers", "-j",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU cores - 1)",
    )
    args = parser.parse_args()

    depth_files = find_depth_pngs(args.root)
    if not depth_files:
        print(f"No depth.png files found under: {args.root}")
        return

    root_resolved = Path(args.root).resolve()
    root_str = str(root_resolved)
    num_workers = args.workers if args.workers is not None else max(1, cpu_count() - 1)
    tasks = [(str(p), root_str) for p in depth_files]

    results = []  # (dataset_name, rel, depth_min, depth_max, depth_mean, valid_ratio)
    failed = []
    total = len(tasks)

    print(f"Scanning {total} depth.png file(s) with {num_workers} worker(s)...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_worker, t): t for t in tasks}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            rel_str, dataset_name, stats = future.result()
            if stats is None:
                failed.append(rel_str)
            else:
                results.append((
                    dataset_name,
                    rel_str,
                    stats["depth_min"],
                    stats["depth_max"],
                    stats["depth_mean"],
                    stats["valid_ratio"],
                ))
                if not args.quiet:
                    print(f"{rel_str}\tmin={stats['depth_min']:.3f}\tmax={stats['depth_max']:.3f}\tmean={stats['depth_mean']:.3f}\tvalid={stats['valid_ratio']*100:.1f}%")
            if completed % 1000 == 0 or completed == total:
                print(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)")

    results.sort(key=lambda r: (r[0], r[1]))

    if failed and not args.quiet:
        print("\nFiles with unreadable/invalid depth:")
        for r in failed:
            print(f"  {r}")

    # Dataset-level summary: overall depth range and mean.
    dataset_agg = defaultdict(lambda: {"min": [], "max": [], "mean": []})
    for r in results:
        dataset_agg[r[0]]["min"].append(r[2])
        dataset_agg[r[0]]["max"].append(r[3])
        dataset_agg[r[0]]["mean"].append(r[4])
    print(f"\nTotal depth.png: {len(depth_files)}, succeeded: {len(results)}, failed: {len(failed)}")
    print("\nDataset\tdepth_min(m)\tdepth_max(m)\tdepth_mean(m)")
    for dataset_name in sorted(dataset_agg.keys()):
        agg = dataset_agg[dataset_name]
        dmin, dmax = min(agg["min"]), max(agg["max"])
        dmean = np.mean(agg["mean"])
        print(f"{dataset_name}\t{dmin:.3f}\t{dmax:.3f}\t{dmean:.3f}")

    vis_dir = get_visualization_dir()

    if args.output and results:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("dataset\trel\tdepth_min\tdepth_max\tdepth_mean\tvalid_ratio\n")
            for r in results:
                f.write(f"{r[0]}\t{r[1]}\t{r[2]:.6f}\t{r[3]:.6f}\t{r[4]:.6f}\t{r[5]:.4f}\n")
        print(f"\nWritten: {out_path}")

    # Depth-count summary by (dataset, rounded depth_mean), saved to CSV.
    if results:
        # (dataset, depth_mean rounded to 3 decimals) -> count
        depth_counts = Counter((r[0], round(r[4], 3)) for r in results)
        print("\n--- Depth (depth_mean) image counts by dataset ---")
        print("Dataset\tdepth_mean(m)\tImage count")
        stats_path = vis_dir / "depth_stats.csv"
        with open(stats_path, "w") as f:
            f.write("dataset,depth_mean,count\n")
            for (dataset_name, dmean) in sorted(depth_counts.keys()):
                c = depth_counts[(dataset_name, dmean)]
                f.write(f"{dataset_name},{dmean:.3f},{c}\n")
                print(f"{dataset_name}\t{dmean:.3f}\t{c}")
        print(f"\nSaved depth stats: {stats_path}")

    # Plot depth distribution: one curve per dataset, x=depth(m), y=ln(count).
    if args.plot and results:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\nmatplotlib is not installed. Install with: pip install matplotlib")
        else:
            # Plot depth range in log scale.
            d_min_plot = 1e-2   # 0.01 m
            d_max_plot = 1e3   # 1000 m
            bins = np.logspace(-3, 3, args.bins + 1)
            bin_centers = (bins[:-1] * bins[1:]) ** 0.5

            dataset_depths = defaultdict(list)
            for dataset_name, _rel, _dmin, _dmax, dmean, _vr in results:
                if d_min_plot <= dmean <= d_max_plot:
                    dataset_depths[dataset_name].append(dmean)

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.tab10(np.linspace(0, 1, max(len(dataset_depths), 1)))
            for idx, (dataset_name, depth_list) in enumerate(sorted(dataset_depths.items())):
                hist, _ = np.histogram(depth_list, bins=bins)
                mask = hist > 0
                x_plot = bin_centers[mask]
                y_plot = np.log(hist[mask].astype(float))
                if len(x_plot) > 0:
                    ax.plot(x_plot, y_plot, "o-", label=dataset_name, color=colors[idx % 10], alpha=0.8, markersize=4)
            ax.set_xscale("log")
            ax.set_xlim(d_min_plot, d_max_plot)
            ax.set_xlabel("Depth (m, mean per image)")
            ax.set_ylabel("ln(Image count)")
            ax.set_title("Depth distribution by dataset")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = Path(args.plot_output) if args.plot_output else vis_dir / "depth_distribution.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Saved depth distribution plot: {plot_path}")


if __name__ == "__main__":
    main()
