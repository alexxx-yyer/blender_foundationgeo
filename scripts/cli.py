#!/usr/bin/env python3
"""渲染与转换 CLI 参数解析"""

import argparse


def add_render_args(parser: argparse.ArgumentParser, include_config: bool = False) -> None:
    parser.add_argument("blend_file", help="输入的 .blend 文件路径")
    parser.add_argument("-o", "--output", required=True, help="输出目录（scene/）")
    parser.add_argument("-c", "--camera", help="相机名称（默认：活动相机）")
    parser.add_argument("-w", "--width", type=int, help="渲染宽度（默认：使用场景设置）")
    parser.add_argument("--height", type=int, help="渲染高度（默认：使用场景设置）")
    parser.add_argument("--export-animation", action="store_true", help="导出动画中每一帧")
    parser.add_argument("--frame-start", type=int, default=None, help="起始帧（默认：使用场景设置）")
    parser.add_argument("--frame-end", type=int, default=None, help="结束帧（默认：使用场景设置）")
    parser.add_argument("--frame-step", type=int, default=1, help="帧步长（默认：1）")
    parser.add_argument("--skip-conversion", action="store_true", help="跳过 EXR 转换（仅渲染）")
    parser.add_argument("--colormap", default="turbo", help="PNG 转换的 colormap（默认：turbo）")
    parser.add_argument("--blender", help="Blender 可执行文件路径（默认：自动查找）")
    parser.add_argument("--device", choices=["CPU", "GPU"], help="渲染设备")
    parser.add_argument("--compute-type",
                        choices=["CUDA", "OPTIX", "HIP", "METAL", "ONEAPI"],
                        help="GPU 计算类型")
    parser.add_argument("--gpu-ids",
                        help="指定使用的 GPU 索引，如 '0,1,2,3' 或 'all'（默认：all）")
    if include_config:
        parser.add_argument("--config", help="YAML 配置文件路径（仅 render 子命令）")


def build_render_parser():
    parser = argparse.ArgumentParser(
        description="整合 Blender 渲染、相机参数导出和 EXR 转换",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 单帧渲染:
  python render_and_convert.py input.blend -o scene/

  # 动画渲染:
  python render_and_convert.py input.blend -o scene/ --export-animation

  # 指定相机和渲染尺寸:
  python render_and_convert.py input.blend -o scene/ -c Camera -w 1920 --height 1080

  # 指定帧范围:
  python render_and_convert.py input.blend -o scene/ --export-animation --frame-start 1 --frame-end 48

  # 跳过 EXR 转换:
  python render_and_convert.py input.blend -o scene/ --skip-conversion
        """,
    )

    add_render_args(parser)

    return parser


def parse_render_args(argv):
    parser = build_render_parser()
    return parser.parse_args(argv)


def build_main_parser():
    parser = argparse.ArgumentParser(
        description="FoundationGeo 工具入口（渲染 + EXR 转换）"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser(
        "render",
        help="调用 Blender 渲染，并可选实时转换 EXR",
    )
    add_render_args(render, include_config=True)

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
    exr2png_cmd.set_defaults(invert=True)
    exr2png_cmd.add_argument("input", help="输入的 EXR 文件路径或目录")
    exr2png_cmd.add_argument("-o", "--output", help="输出的 PNG 文件路径或目录（可选）")
    exr2png_cmd.add_argument("-c", "--colormap", default="turbo",
                             help="Colormap 名称：turbo, turbo_r, viridis, jet, plasma, inferno")
    exr2png_cmd.add_argument("--vmin", type=float, help="Min depth value")
    exr2png_cmd.add_argument("--vmax", type=float, help="Max depth value")
    exr2png_cmd.add_argument("--no-invert", action="store_false", dest="invert",
                             help="Disable invert depth (near=red, far=blue)")
    exr2png_cmd.add_argument("--batch", action="store_true", help="批量处理模式：将输入视为目录")
    exr2png_cmd.add_argument("-r", "--recursive", action="store_true", help="递归搜索子目录（仅批量模式）")

    parallel = subparsers.add_parser(
        "parallel",
        help="多 GPU 并行渲染（每张卡渲染不同帧）",
    )
    parallel.add_argument("blend_file", help="输入的 .blend 文件路径")
    parallel.add_argument("-o", "--output", required=True, help="输出目录")
    parallel.add_argument("--frame-start", type=int, required=True, help="起始帧")
    parallel.add_argument("--frame-end", type=int, required=True, help="结束帧")
    parallel.add_argument("--num-gpus", type=int, default=8, help="使用的 GPU 数量（默认：8）")
    parallel.add_argument("--frame-step", type=int, default=1, help="帧步长（默认：1）")
    parallel.add_argument("--compute-type", default="CUDA",
                          choices=["CUDA", "OPTIX", "HIP", "METAL", "ONEAPI"],
                          help="GPU 计算类型（默认：CUDA）")
    parallel.add_argument("-c", "--camera", help="相机名称")
    parallel.add_argument("-w", "--width", type=int, help="渲染宽度")
    parallel.add_argument("--height", type=int, help="渲染高度")
    parallel.add_argument("--skip-conversion", action="store_true", help="跳过 EXR 转换")
    parallel.add_argument("--colormap", default="turbo", help="PNG colormap")
    parallel.add_argument("--blender", help="Blender 可执行文件路径")

    return parser
