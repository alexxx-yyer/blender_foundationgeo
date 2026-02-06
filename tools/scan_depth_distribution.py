#!/usr/bin/env python3
"""递归扫描目录下所有 depth.png，统计深度分布。

根目录下的一级文件夹视为一个数据集，汇总表第一列为该文件夹名；
可统计每个深度区间对应的图片数量并绘制深度分布图。

depth.png 为 16bit 编码深度图，解码方式与 read_depth.py 一致。

用法:
  python scan_depth_distribution.py /path/to/root   # 如 MoGe/data/eval/ 或 processed/
  python scan_depth_distribution.py /path/to/root --output depths.csv
  python scan_depth_distribution.py /path/to/root -q   # 只输出汇总
  python scan_depth_distribution.py /path/to/root --plot   # 统计深度并画分布图
"""

import argparse
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

# 允许从任意工作目录运行时导入同目录的 read_depth
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

try:
    from read_depth import read_depth_png
except ImportError:
    read_depth_png = None

import numpy as np


def find_depth_pngs(root: str):
    """递归查找 root 下所有 depth.png 路径。"""
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"不是目录: {root}")
    return sorted(root.rglob("depth.png"))


def get_depth_stats(depth_path: str) -> dict | None:
    """从 depth.png 读取并解码，返回该帧的深度统计；失败返回 None。"""
    if read_depth_png is None:
        raise RuntimeError("无法导入 read_depth.read_depth_png，请确保 read_depth.py 在同目录")
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
    """多进程 worker：处理单张 depth.png。返回 (rel_str, dataset_name, stats_dict or None)。"""
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
        description="递归扫描目录下 depth.png，统计深度分布"
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=os.getcwd(),
        help="要扫描的根目录（默认当前目录）",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="可选：将每帧结果写入 CSV（列: dataset,rel,depth_min,depth_max,depth_mean,valid_ratio）",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="只输出有问题的文件或统计，不逐行打印",
    )
    parser.add_argument(
        "--plot", "-p",
        action="store_true",
        help="统计深度分布并绘制分布图",
    )
    parser.add_argument(
        "--plot-output",
        default=None,
        help="分布图保存路径（默认: depth_distribution.png）",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="分布图直方图 bin 数量（默认 50）",
    )
    parser.add_argument(
        "--depth-max",
        type=float,
        default=None,
        help="分布图横轴最大深度（米），超过的归入最后一 bin；默认用数据最大值",
    )
    parser.add_argument(
        "--workers", "-j",
        type=int,
        default=None,
        help="并行进程数（默认: CPU 内核数 - 1）",
    )
    args = parser.parse_args()

    depth_files = find_depth_pngs(args.root)
    if not depth_files:
        print(f"在 {args.root} 下未找到任何 depth.png")
        return

    root_resolved = Path(args.root).resolve()
    root_str = str(root_resolved)
    num_workers = args.workers if args.workers is not None else max(1, cpu_count() - 1)
    tasks = [(str(p), root_str) for p in depth_files]

    results = []  # (dataset_name, rel, depth_min, depth_max, depth_mean, valid_ratio)
    failed = []
    total = len(tasks)

    print(f"扫描 {total} 个 depth.png，使用 {num_workers} 个进程 …")
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
                print(f"进度: {completed}/{total} ({100*completed/total:.1f}%)")

    results.sort(key=lambda r: (r[0], r[1]))

    if failed and not args.quiet:
        print("\n无法解析深度的文件:")
        for r in failed:
            print(f"  {r}")

    # 按数据集汇总：每个数据集的整体深度范围
    dataset_agg = defaultdict(lambda: {"min": [], "max": [], "mean": []})
    for r in results:
        dataset_agg[r[0]]["min"].append(r[2])
        dataset_agg[r[0]]["max"].append(r[3])
        dataset_agg[r[0]]["mean"].append(r[4])
    print(f"\n共 {len(depth_files)} 个 depth.png，成功 {len(results)}，失败 {len(failed)}")
    print("\n数据集\t深度min(m)\t深度max(m)\t深度mean(m)")
    for dataset_name in sorted(dataset_agg.keys()):
        agg = dataset_agg[dataset_name]
        dmin, dmax = min(agg["min"]), max(agg["max"])
        dmean = np.mean(agg["mean"])
        print(f"{dataset_name}\t{dmin:.3f}\t{dmax:.3f}\t{dmean:.3f}")

    if args.output and results:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("dataset\trel\tdepth_min\tdepth_max\tdepth_mean\tvalid_ratio\n")
            for r in results:
                f.write(f"{r[0]}\t{r[1]}\t{r[2]:.6f}\t{r[3]:.6f}\t{r[4]:.6f}\t{r[5]:.4f}\n")
        print(f"\n已写入: {out_path}")

    # 深度统计：按 (数据集, depth_mean) 统计图片数量，保存 CSV（与 fx_stats 一致带数据集）
    if results:
        # (dataset, depth_mean 保留三位小数) -> count
        depth_counts = Counter((r[0], round(r[4], 3)) for r in results)
        print("\n--- 深度 (depth_mean) 图片数量统计（按数据集、depth_mean 排序）---")
        print("数据集\tdepth_mean(m)\t图片数量")
        stats_path = Path(os.getcwd()) / "depth_stats.csv"
        with open(stats_path, "w") as f:
            f.write("dataset,depth_mean,count\n")
            for (dataset_name, dmean) in sorted(depth_counts.keys()):
                c = depth_counts[(dataset_name, dmean)]
                f.write(f"{dataset_name},{dmean:.3f},{c}\n")
                print(f"{dataset_name}\t{dmean:.3f}\t{c}")
        print(f"\n已保存深度统计: {stats_path}")

    # 绘制深度分布图：每个数据集一条曲线，横轴深度(m)，纵轴 ln(数量)
    if args.plot and results:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n未安装 matplotlib，无法绘图。可执行: pip install matplotlib")
        else:
            # 横轴深度限制在 0.01–100 m，以 10 为底对数刻度 (10^-2 到 10^2)
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
            plot_path = Path(args.plot_output) if args.plot_output else Path(os.getcwd()) / "depth_distribution.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"已保存深度分布图: {plot_path}")


if __name__ == "__main__":
    main()
