#!/usr/bin/env python3
"""渲染与转换 CLI 参数解析"""

import argparse


def build_parser():
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

    return parser


def parse_args(argv):
    parser = build_parser()
    return parser.parse_args(argv)
