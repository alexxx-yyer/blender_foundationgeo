#!/usr/bin/env python3
"""CLI argument definitions for rendering and conversion."""

import argparse


def add_render_args(parser: argparse.ArgumentParser, include_config: bool = False) -> None:
    parser.add_argument("blend_file", help="Path to input .blend file")
    parser.add_argument("-o", "--output", required=True, help="Output directory (scene/)")
    parser.add_argument("-c", "--camera", help="Camera name (default: active camera)")
    parser.add_argument("-w", "--width", type=int, help="Render width (default: scene setting)")
    parser.add_argument("--height", type=int, help="Render height (default: scene setting)")
    parser.add_argument("--export-animation", action="store_true", help="Export every frame in animation")
    parser.add_argument("--frame-start", type=int, default=None, help="Start frame (default: scene setting)")
    parser.add_argument("--frame-end", type=int, default=None, help="End frame (default: scene setting)")
    parser.add_argument("--frame-step", type=int, default=1, help="Frame step (default: 1)")
    parser.add_argument("--skip-conversion", action="store_true", help="Skip EXR conversion (render only)")
    parser.add_argument("--colormap", default="turbo", help="Colormap for PNG conversion (default: turbo)")
    parser.add_argument("--blender", help="Path to Blender executable (default: auto-detect)")
    parser.add_argument("--device", choices=["CPU", "GPU"], help="Render device")
    parser.add_argument("--compute-type",
                        choices=["CUDA", "OPTIX", "HIP", "METAL", "ONEAPI"],
                        help="GPU compute backend")
    parser.add_argument("--gpu-ids",
                        help="GPU indices, e.g. '0,1,2,3' or 'all' (default: all)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show verbose Blender output (default: concise progress only)")
    parser.add_argument("--no-compositor", action="store_true",
                        help="Auto-create compositor nodes instead of requiring preconfigured compositor")
    if include_config:
        parser.add_argument("--config", help="Path to YAML config file (render command only)")


def build_render_parser():
    parser = argparse.ArgumentParser(
        description="Render with Blender and convert EXR depth outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single frame:
  python render_and_convert.py input.blend -o scene/

  # Animation:
  python render_and_convert.py input.blend -o scene/ --export-animation

  # Camera and resolution:
  python render_and_convert.py input.blend -o scene/ -c Camera -w 1920 --height 1080

  # Frame range:
  python render_and_convert.py input.blend -o scene/ --export-animation --frame-start 1 --frame-end 48

  # Render only (skip EXR conversion):
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
        description="FoundationGeo tools entrypoint (render + EXR conversion)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser(
        "render",
        help="Run Blender render and optionally convert EXR outputs on the fly",
    )
    add_render_args(render, include_config=True)

    exr2all = subparsers.add_parser(
        "exr2all",
        help="Convert EXR files in depth/exr directory to NPY + PNG",
    )
    exr2all.add_argument("depth_exr_dir", help="Path to depth/exr directory")
    exr2all.add_argument("--colormap", default="turbo", help="Colormap for PNG conversion (default: turbo)")

    exr2npy_cmd = subparsers.add_parser(
        "exr2npy",
        help="Run EXR -> NPY conversion",
    )
    exr2npy_cmd.add_argument("input", help="Input EXR file or directory")
    exr2npy_cmd.add_argument("-o", "--output", help="Output NPY file or directory (optional)")
    exr2npy_cmd.add_argument("--batch", action="store_true", help="Batch mode: treat input as directory")
    exr2npy_cmd.add_argument("-r", "--recursive", action="store_true", help="Search subdirectories recursively (batch mode only)")

    exr2png_cmd = subparsers.add_parser(
        "exr2png",
        help="Run EXR -> PNG conversion",
    )
    exr2png_cmd.set_defaults(invert=True)
    exr2png_cmd.add_argument("input", help="Input EXR file or directory")
    exr2png_cmd.add_argument("-o", "--output", help="Output PNG file or directory (optional)")
    exr2png_cmd.add_argument("-c", "--colormap", default="turbo",
                             help="Colormap name: turbo, turbo_r, viridis, jet, plasma, inferno")
    exr2png_cmd.add_argument("--vmin", type=float, help="Min depth value")
    exr2png_cmd.add_argument("--vmax", type=float, help="Max depth value")
    exr2png_cmd.add_argument("--no-invert", action="store_false", dest="invert",
                             help="Disable invert depth (near=red, far=blue)")
    exr2png_cmd.add_argument("--batch", action="store_true", help="Batch mode: treat input as directory")
    exr2png_cmd.add_argument("-r", "--recursive", action="store_true", help="Search subdirectories recursively (batch mode only)")

    parallel = subparsers.add_parser(
        "parallel",
        help="Multi-GPU parallel render (each GPU renders different frame ranges)",
    )
    parallel.add_argument("blend_file", help="Path to input .blend file")
    parallel.add_argument("-o", "--output", required=True, help="Output directory")
    parallel.add_argument("--frame-start", type=int, required=True, help="Start frame")
    parallel.add_argument("--frame-end", type=int, required=True, help="End frame")
    parallel.add_argument("--num-gpus", type=int, default=None,
                          help="Number of GPUs to use (mutually exclusive with --gpu-ids, default: 8)")
    parallel.add_argument("--gpu-ids", help="GPU indices to use, e.g. '3,4,5,6,7' (mutually exclusive with --num-gpus)")
    parallel.add_argument("--frame-step", type=int, default=1, help="Frame step (default: 1)")
    parallel.add_argument("--compute-type", default="CUDA",
                          choices=["CUDA", "OPTIX", "HIP", "METAL", "ONEAPI"],
                          help="GPU compute backend (default: CUDA)")
    parallel.add_argument("-c", "--camera", help="Camera name")
    parallel.add_argument("-w", "--width", type=int, help="Render width")
    parallel.add_argument("--height", type=int, help="Render height")
    parallel.add_argument("--skip-conversion", action="store_true", help="Skip EXR conversion")
    parallel.add_argument("--colormap", default="turbo", help="PNG colormap")
    parallel.add_argument("--blender", help="Path to Blender executable")
    parallel.add_argument("--no-compositor", action="store_true",
                          help="Auto-create compositor nodes instead of requiring preconfigured compositor")

    return parser
