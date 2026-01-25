#!/usr/bin/env python3
"""统一入口：渲染 + 转换 + 独立转换工具"""

import os
import sys


def _load_modules():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(repo_root, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    import pipeline
    import depth_convert

    return pipeline, depth_convert


def main():
    pipeline, depth_convert = _load_modules()
    import cli
    import config as config_mod
    parser = cli.build_main_parser()
    args = parser.parse_args()

    if args.command == "render":
        config = None
        if args.config:
            config = config_mod.load_yaml_config(args.config)
            args = config_mod.merge_render_config(args, config)
        device = args.device or (config.get("device") if config else None)
        compute_type = args.compute_type or (config.get("compute_type") if config else None)
        gpu_ids = getattr(args, "gpu_ids", None) or (config.get("gpu_ids") if config else None)
        # 如果指定了 compute_type 但没有指定 device，自动设置为 GPU
        if compute_type and not device:
            device = "GPU"
        if device:
            os.environ["FG_DEVICE"] = str(device)
        if compute_type:
            os.environ["FG_COMPUTE_TYPE"] = str(compute_type)
        if gpu_ids:
            os.environ["FG_GPU_IDS"] = str(gpu_ids)
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
        return

    if args.command == "exr2all":
        pipeline.convert_exr_files(args.depth_exr_dir, args.colormap)
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

    if args.command == "parallel":
        import parallel_render
        success = parallel_render.parallel_render(
            args.blend_file,
            args.output,
            args.frame_start,
            args.frame_end,
            args.num_gpus,
            args.frame_step,
            args.compute_type,
            args.camera,
            args.width,
            args.height,
            args.skip_conversion,
            args.colormap,
            args.blender,
        )
        if not success:
            sys.exit(1)
        return

    parser.error("未知命令")


if __name__ == "__main__":
    main()
