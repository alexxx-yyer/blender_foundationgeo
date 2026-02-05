#!/usr/bin/env python3
"""递归扫描目录下所有 meta.json，提取相机内参 fx、fy。

根目录下的一级文件夹视为一个数据集，汇总表第一列为该文件夹名；
同一数据集内相同的 (fx, fy) 只输出一行。
可统计每个 fx 对应的图片数量并绘制 fx 分布图。

meta.json 格式示例:
  {"intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]}

用法:
  python scan_meta_focal.py /path/to/eval   # 如 MoGe/data/eval/
  python scan_meta_focal.py /path/to/root --output focals.csv  # CSV 列为 dataset,fx,fy（制表符分隔）
  python scan_meta_focal.py /path/to/root -q   # 只输出汇总
  python scan_meta_focal.py /path/to/root --plot   # 统计 fx 并画分布图
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


def find_meta_jsons(root: str):
    """递归查找 root 下所有 meta.json 路径。"""
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"不是目录: {root}")
    return sorted(root.rglob("meta.json"))


def get_fx_fy(meta_path: str) -> tuple[float, float] | None:
    """从 meta.json 读取 fx、fy（intrinsics[0][0], intrinsics[1][1]）。无效则返回 None。"""
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
        description="递归扫描目录下 meta.json，提取 fx、fy"
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
        help="可选：将结果写入 CSV（列: dataset,fx,fy，制表符分隔，同数据集同焦距只一行）",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="只输出有问题的文件或统计，不逐行打印",
    )
    parser.add_argument(
        "--plot", "-p",
        action="store_true",
        help="统计每个 fx 的图片数量并绘制 fx 分布图",
    )
    parser.add_argument(
        "--plot-output",
        default=None,
        help="分布图保存路径（默认: fx_distribution.png）",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="分布图直方图 bin 数量（默认 50）",
    )
    args = parser.parse_args()

    meta_files = find_meta_jsons(args.root)
    if not meta_files:
        print(f"在 {args.root} 下未找到任何 meta.json")
        return

    results = []  # (dataset_name, rel, fx, fy)
    failed = []
    root_resolved = Path(args.root).resolve()

    for path in meta_files:
        try:
            rel = path.relative_to(root_resolved)
            rel_str = str(rel)
            # 一级文件夹名作为数据集名
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
        print("\n无法解析 fx/fy 的文件:")
        for r in failed:
            print(f"  {r}")

    # 按数据集汇总：每个一级文件夹为数据集，同一数据集内相同 (fx, fy) 只输出一次
    unique_by_dataset_focal = list(dict.fromkeys((r[0], r[2], r[3]) for r in results))
    unique_by_dataset_focal.sort(key=lambda x: (x[0], x[1], x[2]))
    print(f"\n共 {len(meta_files)} 个 meta.json，成功 {len(results)}，失败 {len(failed)}")
    print("\n数据集\tfx\tfy")
    for dataset_name, fx, fy in unique_by_dataset_focal:
        print(f"{dataset_name}\t{fx:.3f}\t{fy:.3f}")

    if args.output and results:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("dataset\tfx\tfy\n")
            for dataset_name, fx, fy in unique_by_dataset_focal:
                f.write(f"{dataset_name}\t{fx:.3f}\t{fy:.3f}\n")
        print(f"\n已写入: {out_path}")

    # fx 统计：按 (数据集, fx, fy) 统计图片数量，并写入当前目录 CSV
    if results:
        # (dataset, fx, fy) -> count，fx/fy 保留三位小数
        focal_counts = Counter((r[0], round(r[2], 3), round(r[3], 3)) for r in results)
        print("\n--- fx 图片数量统计（按数据集、fx、fy 排序）---")
        print("数据集\tfx\tfy\t图片数量")
        stats_path = Path(os.getcwd()) / "fx_stats.csv"
        with open(stats_path, "w") as f:
            f.write("dataset,fx,fy,count\n")
            for (dataset_name, fx, fy) in sorted(focal_counts.keys()):
                c = focal_counts[(dataset_name, fx, fy)]
                f.write(f"{dataset_name},{fx:.3f},{fy:.3f},{c}\n")
                print(f"{dataset_name}\t{fx:.3f}\t{fy:.3f}\t{c}")
        print(f"\n已保存 fx 统计: {stats_path}")

    # 绘制 fx 分布图：每个数据集一条曲线，共一张图（保存到当前目录，不弹窗显示）
    if args.plot and results:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("\n未安装 matplotlib，无法绘图。可执行: pip install matplotlib")
        else:
            # 按数据集分组 fx
            dataset_fx = defaultdict(list)
            for dataset_name, _rel, fx, _fy in results:
                dataset_fx[dataset_name].append(round(fx, 3))

            # 统一 bins：用全体 fx 范围
            all_fx = [r[2] for r in results]
            fx_min, fx_max = min(all_fx), max(all_fx)
            bins = np.linspace(fx_min, fx_max, args.bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.tab10(np.linspace(0, 1, max(len(dataset_fx), 1)))
            for idx, (dataset_name, fx_list) in enumerate(sorted(dataset_fx.items())):
                hist, _ = np.histogram(fx_list, bins=bins)
                # 只取有图片的 bin，同一数据集内连线成一条曲线；纵轴用 ln(数量)
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
            plot_path = Path(args.plot_output) if args.plot_output else Path(os.getcwd()) / "fx_distribution.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"已保存 fx 分布图: {plot_path}")


if __name__ == "__main__":
    main()
