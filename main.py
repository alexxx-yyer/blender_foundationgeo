#!/usr/bin/env python3
"""统一入口：渲染 + 转换 + 独立转换工具"""

import argparse
import os
import sys


def _load_modules():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(repo_root, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    import render_and_convert
    import exr2npy
    import exr2png

    return render_and_convert, exr2npy, exr2png


def _build_parser():
    parser = argparse.ArgumentParser(
        description="FoundationGeo 工具入口（渲染 + EXR 转换）"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser(
        "render",
        help="调用 Blender 渲染，并可选实时转换 EXR",
    )
    render.add_argument("blend_file", help="输入的 .blend 文件路径")
    render.add_argument("-o", "--output", required=True, help="输出目录（scene/）")
    render.add_argument("-c", "--camera", help="相机名称（默认：活动相机）")
    render.add_argument("-w", "--width", type=int, help="渲染宽度（默认：使用场景设置）")
    render.add_argument("--height", type=int, help="渲染高度（默认：使用场景设置）")
    render.add_argument("--export-animation", action="store_true", help="导出动画中每一帧")
    render.add_argument("--frame-start", type=int, default=None, help="起始帧（默认：使用场景设置）")
    render.add_argument("--frame-end", type=int, default=None, help="结束帧（默认：使用场景设置）")
    render.add_argument("--frame-step", type=int, default=1, help="帧步长（默认：1）")
    render.add_argument("--skip-conversion", action="store_true", help="跳过 EXR 转换（仅渲染）")
    render.add_argument("--colormap", default="turbo", help="PNG 转换的 colormap（默认：turbo）")
    render.add_argument("--blender", help="Blender 可执行文件路径（默认：自动查找）")

    exr2all = subparsers.add_parser(
        "exr2all",
        help="将 depth/exr 目录下的 EXR 转换为 NPY + PNG",
    )
    exr2all.add_argument("depth_exr_dir", help="depth/exr 目录路径")
    exr2all.add_argument("--colormap", default="turbo", help="PNG 转换的 colormap（默认：turbo）")

    exr2npy_cmd = subparsers.add_parser(
        "exr2npy",
        help="单独执行 EXR -> NPY",
    )
    exr2npy_cmd.add_argument("input", help="输入的 EXR 文件路径或目录")
    exr2npy_cmd.add_argument("-o", "--output", help="输出的 NPY 文件路径或目录（可选）")
    exr2npy_cmd.add_argument("--batch", action="store_true", help="批量处理模式：将输入视为目录")
    exr2npy_cmd.add_argument("-r", "--recursive", action="store_true", help="递归搜索子目录（仅批量模式）")

    exr2png_cmd = subparsers.add_parser(
        "exr2png",
        help="单独执行 EXR -> PNG",
    )
    exr2png_cmd.add_argument("input", help="输入的 EXR 文件路径或目录")
    exr2png_cmd.add_argument("-o", "--output", help="输出的 PNG 文件路径或目录（可选）")
    exr2png_cmd.add_argument("-c", "--colormap", default="turbo",
                             help="Colormap 名称：turbo, turbo_r, viridis, jet, plasma, inferno")
    exr2png_cmd.add_argument("--vmin", type=float, help="Min depth value")
    exr2png_cmd.add_argument("--vmax", type=float, help="Max depth value")
    exr2png_cmd.add_argument("-i", "--invert", action="store_true",
                             help="Invert depth (near=red, far=blue)")
    exr2png_cmd.add_argument("--batch", action="store_true", help="批量处理模式：将输入视为目录")
    exr2png_cmd.add_argument("-r", "--recursive", action="store_true", help="递归搜索子目录（仅批量模式）")

    return parser


def main():
    render_and_convert, exr2npy, exr2png = _load_modules()
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "render":
        ok = render_and_convert.main_external(
            args.blend_file,
            args.output,
            args.camera,
            args.width,
            args.height,
            args.export_animation,
            args.frame_start,
            args.frame_end,
            args.frame_step,
            args.skip_conversion,
            args.colormap,
            args.blender,
        )
        if not ok:
            sys.exit(1)
        return

    if args.command == "exr2all":
        render_and_convert.convert_exr_files(args.depth_exr_dir, args.colormap)
        return

    if args.command == "exr2npy":
        input_path = os.path.expanduser(args.input)
        if args.batch or os.path.isdir(input_path):
            exr2npy.batch_exr_to_npy(input_path, args.output, args.recursive)
        else:
            exr2npy.exr_to_npy(args.input, args.output)
        return

    if args.command == "exr2png":
        input_path = os.path.expanduser(args.input)
        if args.batch or os.path.isdir(input_path):
            exr2png.batch_exr_to_png(
                input_path,
                args.output,
                args.colormap,
                args.vmin,
                args.vmax,
                args.invert,
                args.recursive,
            )
        else:
            exr2png.exr_to_png(
                args.input,
                args.output,
                args.colormap,
                args.vmin,
                args.vmax,
                args.invert,
            )
        return

    parser.error("未知命令")


if __name__ == "__main__":
    main()
