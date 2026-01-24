#!/usr/bin/env python3
"""统一入口：渲染 + 转换 + 独立转换工具"""

import os
import sys


def _load_modules():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(repo_root, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    import render_and_convert
    import depth_convert

    return render_and_convert, depth_convert


def _load_yaml_config(path: str) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("缺少 PyYAML，请先安装：pip install pyyaml") from exc

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("YAML 顶层必须是字典")
    return data


def _merge_render_config(args, config: dict):
    if not config:
        return args

    def pick(value, default):
        return value if value is not None else default

    args.blend_file = pick(getattr(args, "blend_file", None), config.get("input"))
    args.output = pick(getattr(args, "output", None), config.get("output"))
    args.camera = pick(getattr(args, "camera", None), config.get("camera"))
    args.width = pick(getattr(args, "width", None), config.get("width"))
    args.height = pick(getattr(args, "height", None), config.get("height"))

    if not args.export_animation and config.get("export_animation") is True:
        args.export_animation = True
    if args.frame_start is None:
        args.frame_start = config.get("frame_start")
    if args.frame_end is None:
        args.frame_end = config.get("frame_end")
    if args.frame_step == 1 and config.get("frame_step") not in (None, 1):
        args.frame_step = config.get("frame_step")

    if not args.skip_conversion and config.get("skip_conversion") is True:
        args.skip_conversion = True
    if args.colormap == "turbo" and config.get("colormap"):
        args.colormap = config.get("colormap")
    if args.blender is None:
        args.blender = config.get("blender")

    return args


def main():
    render_and_convert, depth_convert = _load_modules()
    import cli
    parser = cli.build_main_parser()
    args = parser.parse_args()

    if args.command == "render":
        config = None
        if args.config:
            config = _load_yaml_config(args.config)
            args = _merge_render_config(args, config)
        device = args.device or (config.get("device") if config else None)
        compute_type = args.compute_type or (config.get("compute_type") if config else None)
        if device:
            os.environ["FG_DEVICE"] = str(device)
        if compute_type:
            os.environ["FG_COMPUTE_TYPE"] = str(compute_type)
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
            depth_convert.batch_exr_to_npy(input_path, args.output, args.recursive)
        else:
            depth_convert.exr_to_npy(args.input, args.output)
        return

    if args.command == "exr2png":
        input_path = os.path.expanduser(args.input)
        if args.batch or os.path.isdir(input_path):
            depth_convert.batch_exr_to_png(
                input_path,
                args.output,
                args.colormap,
                args.vmin,
                args.vmax,
                args.invert,
                args.recursive,
            )
        else:
            depth_convert.exr_to_png(
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
