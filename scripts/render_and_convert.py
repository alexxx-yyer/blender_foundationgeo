#!/usr/bin/env python3
"""整合 Blender 渲染、相机参数导出和 EXR 转换功能（CLI 入口）"""

import os
import sys

import cli
import pipeline


if __name__ == "__main__":
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = sys.argv[1:]

    args = cli.parse_render_args(argv)

    if args.device:
        os.environ["FG_DEVICE"] = str(args.device)
    if args.compute_type:
        os.environ["FG_COMPUTE_TYPE"] = str(args.compute_type)

    if pipeline.IN_BLENDER:
        try:
            pipeline.render_and_export(
                args.blend_file,
                args.output,
                args.camera,
                args.width,
                args.height,
                args.export_animation,
                args.frame_start,
                args.frame_end,
                args.frame_step,
            )
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            ok = pipeline.main_external(
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
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)
