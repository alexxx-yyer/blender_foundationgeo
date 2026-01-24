#!/usr/bin/env python3
"""EXR 深度转换（EXR -> NPY / PNG）"""

import contextlib
import io
import glob
import os


try:
    from exr2npy import exr_to_npy
    from exr2png import exr_to_png
except ImportError as e:
    print(f"警告: 无法导入转换函数: {e}")
    exr_to_npy = None
    exr_to_png = None


def convert_single_exr(exr_file: str, depth_exr_dir: str, colormap: str = "turbo",
                       silent: bool = True) -> bool:
    """
    转换单个 EXR 文件为 NPY 和 PNG
    """
    if exr_to_npy is None or exr_to_png is None:
        if not silent:
            print("警告: EXR 转换函数不可用，跳过转换")
        return False

    exr_file = os.path.abspath(os.path.expanduser(exr_file))

    if not os.path.exists(exr_file):
        if not silent:
            print(f"警告: 文件不存在: {exr_file}")
        return False

    try:
        base_name = os.path.basename(exr_file)
        base_name_no_ext = os.path.splitext(base_name)[0]

        # 创建输出目录
        depth_npy_dir = os.path.join(os.path.dirname(depth_exr_dir), "npy")
        depth_vis_dir = os.path.join(os.path.dirname(depth_exr_dir), "vis")
        os.makedirs(depth_npy_dir, exist_ok=True)
        os.makedirs(depth_vis_dir, exist_ok=True)

        if silent:
            # 重定向 stdout 以抑制输出
            with contextlib.redirect_stdout(io.StringIO()):
                npy_path = os.path.join(depth_npy_dir, f"{base_name_no_ext}.npy")
                exr_to_npy(exr_file, npy_path)

                png_path = os.path.join(depth_vis_dir, f"{base_name_no_ext}.png")
                exr_to_png(exr_file, png_path, colormap=colormap)
        else:
            npy_path = os.path.join(depth_npy_dir, f"{base_name_no_ext}.npy")
            exr_to_npy(exr_file, npy_path)

            png_path = os.path.join(depth_vis_dir, f"{base_name_no_ext}.png")
            exr_to_png(exr_file, png_path, colormap=colormap)
            print(f"  ✓ 转换完成: {base_name}")

        return True
    except Exception as e:
        if not silent:
            print(f"  ✗ 转换失败 {os.path.basename(exr_file)}: {e}")
        return False


def convert_exr_files(depth_exr_dir: str, colormap: str = "turbo") -> None:
    """
    将 depth/exr/ 目录中的 EXR 文件转换为 NPY 和 PNG
    """
    if exr_to_npy is None or exr_to_png is None:
        print("警告: EXR 转换函数不可用，跳过转换")
        return

    depth_exr_dir = os.path.expanduser(depth_exr_dir)

    if not os.path.isdir(depth_exr_dir):
        print(f"警告: 目录不存在: {depth_exr_dir}")
        return

    exr_files = sorted(glob.glob(os.path.join(depth_exr_dir, "*.exr")))

    if not exr_files:
        print(f"在目录 {depth_exr_dir} 中未找到 EXR 文件")
        return

    print(f"\n找到 {len(exr_files)} 个 EXR 文件，开始转换...")

    depth_npy_dir = os.path.join(os.path.dirname(depth_exr_dir), "npy")
    depth_vis_dir = os.path.join(os.path.dirname(depth_exr_dir), "vis")
    os.makedirs(depth_npy_dir, exist_ok=True)
    os.makedirs(depth_vis_dir, exist_ok=True)

    success_count = 0
    fail_count = 0

    for exr_file in exr_files:
        try:
            base_name = os.path.basename(exr_file)
            base_name_no_ext = os.path.splitext(base_name)[0]

            npy_path = os.path.join(depth_npy_dir, f"{base_name_no_ext}.npy")
            exr_to_npy(exr_file, npy_path)

            png_path = os.path.join(depth_vis_dir, f"{base_name_no_ext}.png")
            exr_to_png(exr_file, png_path, colormap=colormap)

            success_count += 1
        except Exception as e:
            print(f"  ✗ 转换失败 {os.path.basename(exr_file)}: {e}")
            fail_count += 1

    print(f"\n转换完成!")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  NPY: {depth_npy_dir}")
    print(f"  PNG: {depth_vis_dir}")
