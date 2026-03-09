#!/usr/bin/env python3
"""Depth conversion utilities (EXR -> NPY / PNG / NPY+PNG)."""

import argparse
import contextlib
import glob
import io
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
import OpenEXR
import Imath
import matplotlib.pyplot as plt


def exr_to_npy(exr_path: str, npy_path: str | None = None) -> np.ndarray:
    """
    Convert an EXR file to NPY format.
    """
    exr_path = os.path.expanduser(exr_path)

    if not os.path.exists(exr_path):
        raise FileNotFoundError(f"File not found: {exr_path}")

    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()

    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    channels = list(header["channels"].keys())

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    channel_data = {}
    for channel in channels:
        raw_data = exr_file.channel(channel, pt)
        channel_data[channel] = np.frombuffer(raw_data, dtype=np.float32).reshape(height, width)

    if "R" in channels and "G" in channels and "B" in channels:
        if "A" in channels:
            img = np.stack(
                [channel_data["R"], channel_data["G"], channel_data["B"], channel_data["A"]],
                axis=-1,
            )
        else:
            img = np.stack([channel_data["R"], channel_data["G"], channel_data["B"]], axis=-1)
    elif "Y" in channels:
        img = channel_data["Y"]
    else:
        sorted_channels = sorted(channels)
        if len(sorted_channels) == 1:
            img = channel_data[sorted_channels[0]]
        else:
            img = np.stack([channel_data[c] for c in sorted_channels], axis=-1)

    if npy_path is None:
        npy_path = os.path.splitext(exr_path)[0] + ".npy"
    else:
        npy_path = os.path.expanduser(npy_path)
        # If output is a directory, create output with the same basename.
        if os.path.isdir(npy_path) or npy_path.endswith(os.sep):
            os.makedirs(npy_path, exist_ok=True)
            basename = os.path.splitext(os.path.basename(exr_path))[0] + ".npy"
            npy_path = os.path.join(npy_path, basename)

    os.makedirs(os.path.dirname(npy_path) or ".", exist_ok=True)
    np.save(npy_path, img)

    print("Conversion complete!")
    print(f"  Input: {exr_path}")
    print(f"  Output: {npy_path}")
    print(f"  Shape: {img.shape}")
    print(f"  Data type: {img.dtype}")
    print(f"  Channels: {channels}")

    return img


def exr_to_png(exr_path: str, png_path: str | None = None, colormap: str = "viridis",
               vmin: float | None = None, vmax: float | None = None,
               invert: bool = True) -> np.ndarray:
    """
    Convert an EXR depth file to a pseudocolor PNG.
    """
    exr_path = os.path.expanduser(exr_path)

    if not os.path.exists(exr_path):
        raise FileNotFoundError(f"File not found: {exr_path}")

    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()

    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    channels = list(header["channels"].keys())

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    channel_data = {}
    for channel in channels:
        raw_data = exr_file.channel(channel, pt)
        channel_data[channel] = np.frombuffer(raw_data, dtype=np.float32).reshape(height, width)

    if len(channels) == 1:
        depth = channel_data[channels[0]]
    elif "Z" in channels:
        depth = channel_data["Z"]
    elif "depth" in channels:
        depth = channel_data["depth"]
    else:
        depth = channel_data[sorted(channels)[0]]

    # Filter Blender infinite-depth background values.
    INF_THRESHOLD = 1e9
    valid_mask = depth < INF_THRESHOLD

    if vmin is None:
        vmin = np.min(depth[valid_mask]) if np.any(valid_mask) else np.min(depth)
    if vmax is None:
        vmax = np.max(depth[valid_mask]) if np.any(valid_mask) else np.max(depth)

    depth_normalized = (depth - vmin) / (vmax - vmin + 1e-8)
    depth_normalized = np.clip(depth_normalized, 0, 1)

    if invert:
        depth_normalized = 1.0 - depth_normalized

    cmap = plt.colormaps[colormap]
    colored = cmap(depth_normalized)

    rgb = (colored[:, :, :3] * 255).astype(np.uint8)

    if png_path is None:
        png_path = os.path.splitext(exr_path)[0] + ".png"
    else:
        png_path = os.path.expanduser(png_path)
        # If output is a directory, create output with the same basename.
        if os.path.isdir(png_path) or png_path.endswith(os.sep):
            os.makedirs(png_path, exist_ok=True)
            basename = os.path.splitext(os.path.basename(exr_path))[0] + ".png"
            png_path = os.path.join(png_path, basename)

    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
    plt.imsave(png_path, rgb)

    print("Done!")
    print(f"  Input: {exr_path}")
    print(f"  Output: {png_path}")
    print(f"  Shape: {depth.shape}")
    print(f"  Depth range: [{vmin:.4f}, {vmax:.4f}]")
    print(f"  Colormap: {colormap}")
    print(f"  Inverted: {invert}")
    print(f"  Channels: {channels}")

    return depth


def batch_exr_to_npy(input_dir: str, output_dir: str | None = None, recursive: bool = False):
    """Batch-convert EXR files in a directory to NPY."""
    input_dir = os.path.expanduser(input_dir)

    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Directory does not exist: {input_dir}")

    if recursive:
        pattern = os.path.join(input_dir, "**", "*.exr")
        exr_files = glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(input_dir, "*.exr")
        exr_files = glob.glob(pattern)

    if not exr_files:
        print(f"No EXR files found in directory: {input_dir}")
        return

    print(f"Found {len(exr_files)} EXR file(s)")

    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = os.path.expanduser(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    success_count = 0
    fail_count = 0

    for exr_file in sorted(exr_files):
        try:
            if output_dir == input_dir:
                npy_path = None
            else:
                rel_path = os.path.relpath(exr_file, input_dir)
                rel_dir = os.path.dirname(rel_path)
                base_name = os.path.splitext(os.path.basename(rel_path))[0]

                if rel_dir:
                    output_subdir = os.path.join(output_dir, rel_dir)
                    os.makedirs(output_subdir, exist_ok=True)
                    npy_path = os.path.join(output_subdir, base_name + ".npy")
                else:
                    npy_path = os.path.join(output_dir, base_name + ".npy")

            print(f"\nProcessing: {os.path.basename(exr_file)}")
            exr_to_npy(exr_file, npy_path)
            success_count += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            fail_count += 1

    print("\nBatch conversion complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")


def batch_exr_to_png(input_dir: str, output_dir: str | None = None,
                     colormap: str = "turbo", vmin: float | None = None,
                     vmax: float | None = None, invert: bool = True,
                     recursive: bool = False):
    """Batch-convert EXR files in a directory to PNG."""
    input_dir = os.path.expanduser(input_dir)

    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Directory does not exist: {input_dir}")

    if recursive:
        pattern = os.path.join(input_dir, "**", "*.exr")
        exr_files = glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(input_dir, "*.exr")
        exr_files = glob.glob(pattern)

    if not exr_files:
        print(f"No EXR files found in directory: {input_dir}")
        return

    print(f"Found {len(exr_files)} EXR file(s)")

    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = os.path.expanduser(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    success_count = 0
    fail_count = 0

    for exr_file in sorted(exr_files):
        try:
            if output_dir == input_dir:
                png_path = None
            else:
                rel_path = os.path.relpath(exr_file, input_dir)
                rel_dir = os.path.dirname(rel_path)
                base_name = os.path.splitext(os.path.basename(rel_path))[0]

                if rel_dir:
                    output_subdir = os.path.join(output_dir, rel_dir)
                    os.makedirs(output_subdir, exist_ok=True)
                    png_path = os.path.join(output_subdir, base_name + ".png")
                else:
                    png_path = os.path.join(output_dir, base_name + ".png")

            print(f"\nProcessing: {os.path.basename(exr_file)}")
            exr_to_png(exr_file, png_path, colormap, vmin, vmax, invert)
            success_count += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            fail_count += 1

    print("\nBatch conversion complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")


def convert_single_exr(exr_file: str, depth_exr_dir: str, colormap: str = "turbo",
                       silent: bool = True) -> bool:
    """Convert one EXR file into both NPY and PNG outputs."""
    exr_file = os.path.abspath(os.path.expanduser(exr_file))

    if not os.path.exists(exr_file):
        if not silent:
            print(f"Warning: file does not exist: {exr_file}")
        return False

    try:
        base_name = os.path.basename(exr_file)
        base_name_no_ext = os.path.splitext(base_name)[0]

        depth_npy_dir = os.path.join(os.path.dirname(depth_exr_dir), "npy")
        depth_vis_dir = os.path.join(os.path.dirname(depth_exr_dir), "vis")
        os.makedirs(depth_npy_dir, exist_ok=True)
        os.makedirs(depth_vis_dir, exist_ok=True)

        if silent:
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
            print(f"  ✓ Converted: {base_name}")

        return True
    except Exception as e:
        if not silent:
            print(f"  ✗ Conversion failed for {os.path.basename(exr_file)}: {e}")
        return False


def _convert_single_exr_worker(args: tuple) -> tuple:
    """Multiprocessing worker for single EXR conversion."""
    exr_file, depth_npy_dir, depth_vis_dir, colormap = args
    try:
        base_name = os.path.basename(exr_file)
        base_name_no_ext = os.path.splitext(base_name)[0]

        # Silent conversion
        with contextlib.redirect_stdout(io.StringIO()):
            npy_path = os.path.join(depth_npy_dir, f"{base_name_no_ext}.npy")
            exr_to_npy(exr_file, npy_path)

            png_path = os.path.join(depth_vis_dir, f"{base_name_no_ext}.png")
            exr_to_png(exr_file, png_path, colormap=colormap)

        return (True, base_name, None)
    except Exception as e:
        return (False, os.path.basename(exr_file), str(e))


def convert_exr_files(depth_exr_dir: str, colormap: str = "turbo", 
                      num_workers: int = None) -> None:
    """Convert EXR files under depth/exr/ to NPY and PNG (multiprocessing)."""
    depth_exr_dir = os.path.expanduser(depth_exr_dir)

    if not os.path.isdir(depth_exr_dir):
        print(f"Warning: directory does not exist: {depth_exr_dir}")
        return

    exr_files = sorted(glob.glob(os.path.join(depth_exr_dir, "*.exr")))

    if not exr_files:
        print(f"No EXR files found in directory: {depth_exr_dir}")
        return

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Reserve one core for the OS.

    print(f"\nFound {len(exr_files)} EXR file(s), converting with {num_workers} worker(s)...")

    depth_npy_dir = os.path.join(os.path.dirname(depth_exr_dir), "npy")
    depth_vis_dir = os.path.join(os.path.dirname(depth_exr_dir), "vis")
    os.makedirs(depth_npy_dir, exist_ok=True)
    os.makedirs(depth_vis_dir, exist_ok=True)

    # Build task arguments.
    tasks = [(exr_file, depth_npy_dir, depth_vis_dir, colormap) for exr_file in exr_files]

    success_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_convert_single_exr_worker, task): task for task in tasks}
        
        completed = 0
        total = len(futures)
        for future in as_completed(futures):
            success, filename, error = future.result()
            completed += 1
            if success:
                success_count += 1
            else:
                fail_count += 1
                print(f"  ✗ Conversion failed for {filename}: {error}")
            
            # Print progress.
            if completed % 100 == 0 or completed == total:
                print(f"  Progress: {completed}/{total} ({100*completed/total:.1f}%)")

    print("\nConversion complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  NPY: {depth_npy_dir}")
    print(f"  PNG: {depth_vis_dir}")


def _build_parser():
    parser = argparse.ArgumentParser(
        description="EXR depth conversion (NPY/PNG/NPY+PNG)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single EXR -> NPY
  python depth_convert.py exr2npy input.exr

  # Convert a single EXR -> PNG
  python depth_convert.py exr2png input.exr -c turbo

  # Batch-convert EXR files in a directory -> NPY + PNG
  python depth_convert.py exr2all /path/to/depth/exr
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    exr2npy_cmd = subparsers.add_parser("exr2npy", help="EXR -> NPY")
    exr2npy_cmd.add_argument("input", help="Input EXR file or directory")
    exr2npy_cmd.add_argument("-o", "--output", help="Output NPY file or directory (optional)")
    exr2npy_cmd.add_argument("--batch", action="store_true", help="Batch mode: treat input as directory")
    exr2npy_cmd.add_argument("-r", "--recursive", action="store_true", help="Search subdirectories recursively (batch mode only)")

    exr2png_cmd = subparsers.add_parser("exr2png", help="EXR -> PNG")
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

    exr2all_cmd = subparsers.add_parser("exr2all", help="EXR -> NPY + PNG")
    exr2all_cmd.add_argument("depth_exr_dir", help="Path to depth/exr directory")
    exr2all_cmd.add_argument("--colormap", default="turbo", help="Colormap for PNG conversion (default: turbo)")

    return parser


def _main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "exr2npy":
        input_path = os.path.expanduser(args.input)
        if args.batch or os.path.isdir(input_path):
            batch_exr_to_npy(input_path, args.output, args.recursive)
        else:
            exr_to_npy(args.input, args.output)
        return

    if args.command == "exr2png":
        input_path = os.path.expanduser(args.input)
        if args.batch or os.path.isdir(input_path):
            batch_exr_to_png(
                input_path,
                args.output,
                args.colormap,
                args.vmin,
                args.vmax,
                args.invert,
                args.recursive,
            )
        else:
            exr_to_png(
                args.input,
                args.output,
                args.colormap,
                args.vmin,
                args.vmax,
                args.invert,
            )
        return

    if args.command == "exr2all":
        convert_exr_files(args.depth_exr_dir, args.colormap)
        return

    parser.error("Unknown command")


if __name__ == "__main__":
    _main()
