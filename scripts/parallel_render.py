#!/usr/bin/env python3
"""Multiprocess parallel rendering: each GPU renders a separate frame range.

Each subprocess is pinned to one GPU via CUDA_VISIBLE_DEVICES and FG_GPU_IDS=0
so Blender initializes on that card instead of concentrating work on GPU 0.
"""

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def get_blend_info(blend_file: str, blender_exe: str) -> dict:
    """Read render metadata from a .blend file (launches Blender once)."""
    script = '''
import bpy
import json
import sys

scene = bpy.context.scene
info = {
    "width": scene.render.resolution_x,
    "height": scene.render.resolution_y,
    "samples": getattr(scene.cycles, "samples", None) if hasattr(scene, "cycles") else None,
    "engine": scene.render.engine,
    "frame_start": scene.frame_start,
    "frame_end": scene.frame_end,
}
print("BLEND_INFO:" + json.dumps(info))
'''
    try:
        result = subprocess.run(
            [blender_exe, "--background", blend_file, "--python-expr", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        for line in result.stdout.split("\n"):
            if line.startswith("BLEND_INFO:"):
                return json.loads(line[len("BLEND_INFO:"):])
    except Exception:
        pass
    return {}


def find_blender_executable():
    """Find Blender executable."""
    possible_paths = [
        # Common user-local install paths.
        os.path.expanduser("~/blender-4.2.17-linux-x64/blender"),
        os.path.expanduser("~/blender-4.2.0-linux-x64/blender"),
        os.path.expanduser("~/blender-3.6.5-linux-x64/blender"),
        # System paths.
        "blender",
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "/opt/blender/blender",
    ]

    for path in possible_paths:
        try:
            result = subprocess.run(
                [path, "--version"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                return path
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    return None


def render_worker(args: dict) -> dict:
    """Single render worker process."""
    gpu_id = args["gpu_id"]
    frame_start = args["frame_start"]
    frame_end = args["frame_end"]
    blend_file = args["blend_file"]
    output_dir = args["output_dir"]
    blender_exe = args["blender_exe"]
    compute_type = args.get("compute_type", "CUDA")
    frame_step = args.get("frame_step", 1)
    camera = args.get("camera")
    width = args.get("width")
    height = args.get("height")
    skip_conversion = args.get("skip_conversion", False)
    colormap = args.get("colormap", "turbo")
    use_compositor = args.get("use_compositor", True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "render_and_convert.py")

    # Expose exactly one GPU to the subprocess and force Blender to use it as device 0.
    env = os.environ.copy()
    env["FG_DEVICE"] = "GPU"
    env["FG_COMPUTE_TYPE"] = compute_type
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["FG_GPU_IDS"] = "0"
    env["FG_USE_COMPOSITOR"] = "1" if use_compositor else "0"

    cmd = [
        blender_exe,
        "--background",
        "--python", script_path,
        "--",
        blend_file,
        "--output", output_dir,
        "--export-animation",
        "--frame-start", str(frame_start),
        "--frame-end", str(frame_end),
        "--frame-step", str(frame_step),
    ]

    if camera:
        cmd.extend(["--camera", camera])
    if width:
        cmd.extend(["--width", str(width)])
    if height:
        cmd.extend(["--height", str(height)])
    if skip_conversion:
        cmd.append("--skip-conversion")
    if colormap:
        cmd.extend(["--colormap", colormap])
    if not use_compositor:
        cmd.append("--no-compositor")

    print(f"[GPU {gpu_id}] Start rendering frames {frame_start}-{frame_end}")
    sys.stdout.flush()

    try:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        stderr_output = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                line = line.rstrip()
                if line:
                    # Show progress and errors only.
                    is_progress = line.startswith("Render progress")
                    is_error = any(kw in line for kw in ["Error", "Warning", "Traceback", "Exception"])
                    
                    if is_progress:
                        # Print progress on its own line.
                        print(f"[GPU {gpu_id}] {line}")
                        sys.stdout.flush()
                    elif is_error:
                        print(f"[GPU {gpu_id}] {line}")
                        sys.stdout.flush()
                        stderr_output.append(line)

        returncode = process.wait()

        if returncode == 0:
            print(f"[GPU {gpu_id}] Completed frames {frame_start}-{frame_end}")
            return {
                "gpu_id": gpu_id,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "success": True,
            }
        else:
            error_msg = "\n".join(stderr_output[-5:]) if stderr_output else "Unknown error"
            print(f"[GPU {gpu_id}] Failed: {error_msg[:200]}")
            return {
                "gpu_id": gpu_id,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "success": False,
                "error": error_msg,
            }
    except Exception as e:
        print(f"[GPU {gpu_id}] Exception: {e}")
        return {
            "gpu_id": gpu_id,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "success": False,
            "error": str(e),
        }


def distribute_frames(frame_start: int, frame_end: int, num_gpus: int, frame_step: int = 1,
                      gpu_ids: list = None):
    """Distribute frame ranges across multiple GPUs.
    
    Args:
        frame_start: Start frame.
        frame_end: End frame.
        num_gpus: Number of GPUs (used when gpu_ids is None).
        frame_step: Frame step.
        gpu_ids: Explicit GPU indices, e.g. [3, 4, 5, 6, 7].
    """
    frames = list(range(frame_start, frame_end + 1, frame_step))
    total_frames = len(frames)
    
    # Determine GPU list to use.
    if gpu_ids is not None:
        actual_gpus = gpu_ids
    else:
        actual_gpus = list(range(num_gpus))
    
    actual_num_gpus = len(actual_gpus)
    frames_per_gpu = math.ceil(total_frames / actual_num_gpus)

    distributions = []
    for i, gpu_id in enumerate(actual_gpus):
        start_idx = i * frames_per_gpu
        end_idx = min(start_idx + frames_per_gpu, total_frames)

        if start_idx >= total_frames:
            break

        gpu_frames = frames[start_idx:end_idx]
        if gpu_frames:
            distributions.append({
                "gpu_id": gpu_id,
                "frame_start": gpu_frames[0],
                "frame_end": gpu_frames[-1],
            })

    return distributions


def get_visible_gpus():
    """Read visible GPU list from CUDA_VISIBLE_DEVICES."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_visible:
        return None
    try:
        return [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
    except ValueError:
        return None


def parallel_render(
    blend_file: str,
    output_dir: str,
    frame_start: int,
    frame_end: int,
    num_gpus: int = 8,
    gpu_ids: list = None,
    frame_step: int = 1,
    compute_type: str = "CUDA",
    camera: str = None,
    width: int = None,
    height: int = None,
    skip_conversion: bool = False,
    colormap: str = "turbo",
    blender_exe: str = None,
    use_compositor: bool = True,
):
    """Parallel multi-GPU rendering.

    GPU priority: --gpu-ids > CUDA_VISIBLE_DEVICES > --num-gpus.
    """
    # Determine GPU list to use.
    if gpu_ids is None:
        # Try environment variable first.
        env_gpus = get_visible_gpus()
        if env_gpus is not None:
            gpu_ids = env_gpus
            print(f"Detected GPUs from CUDA_VISIBLE_DEVICES: {gpu_ids}")
        else:
            gpu_ids = list(range(num_gpus))

    if blender_exe is None:
        blender_exe = find_blender_executable()
        if blender_exe is None:
            raise RuntimeError("Blender not found. Use --blender to specify the executable path.")
    else:
        blender_exe = os.path.expanduser(blender_exe)

    blend_file = os.path.expanduser(blend_file)
    output_dir = os.path.expanduser(output_dir)

    if not os.path.exists(blend_file):
        raise FileNotFoundError(f"File not found: {blend_file}")

    os.makedirs(output_dir, exist_ok=True)

    print("Reading scene metadata...")
    blend_info = get_blend_info(blend_file, blender_exe)

    # Use CLI overrides if set, otherwise use .blend values.
    render_width = width if width else blend_info.get("width", "unknown")
    render_height = height if height else blend_info.get("height", "unknown")
    samples = blend_info.get("samples", "unknown")
    engine = blend_info.get("engine", "unknown")

    distributions = distribute_frames(frame_start, frame_end, num_gpus, frame_step, gpu_ids)
    
    # Actual GPU list used.
    actual_gpus = [d["gpu_id"] for d in distributions]

    print(f"\n{'=' * 60}")
    print("Multi-GPU Parallel Rendering")
    print(f"{'=' * 60}")
    print(f"  Blender: {blender_exe}")
    print(f"  Input file: {blend_file}")
    print(f"  Output dir: {output_dir}")
    print(f"  Render engine: {engine}")
    print(f"  Resolution: {render_width} x {render_height}")
    print(f"  Samples: {samples}")
    print(f"  Frame range: {frame_start} - {frame_end} (step: {frame_step})")
    print(f"  GPUs: {','.join(map(str, actual_gpus))} (count: {len(actual_gpus)})")
    print(f"  Compute type: {compute_type}")
    print("\nFrame distribution:")
    for dist in distributions:
        frames_count = len(range(dist["frame_start"], dist["frame_end"] + 1, frame_step))
        print(f"  GPU {dist['gpu_id']}: frames {dist['frame_start']}-{dist['frame_end']} ({frames_count} frame(s))")
    print(f"{'=' * 60}\n")
    sys.stdout.flush()

    tasks = []
    for dist in distributions:
        tasks.append({
            "gpu_id": dist["gpu_id"],
            "frame_start": dist["frame_start"],
            "frame_end": dist["frame_end"],
            "blend_file": blend_file,
            "output_dir": output_dir,
            "blender_exe": blender_exe,
            "compute_type": compute_type,
            "frame_step": frame_step,
            "camera": camera,
            "width": width,
            "height": height,
            "skip_conversion": skip_conversion,
            "colormap": colormap,
            "use_compositor": use_compositor,
        })

    results = []
    with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        futures = {executor.submit(render_worker, task): task for task in tasks}

        # Wait for all tasks; subprocesses stream progress.
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count

    print(f"\n{'=' * 60}")
    print("Render finished!")
    print(f"{'=' * 60}")
    print(f"  Success: {success_count}/{len(results)}")
    if fail_count > 0:
        print(f"  Failed: {fail_count}")
        for r in results:
            if not r["success"]:
                print(f"    - GPU {r['gpu_id']}: frames {r['frame_start']}-{r['frame_end']}")
    print(f"{'=' * 60}")

    # EXR conversion.
    if not skip_conversion and success_count > 0:
        depth_exr_dir = os.path.join(output_dir, "depth", "exr")
        if os.path.isdir(depth_exr_dir):
            print(f"\n{'=' * 60}")
            print("Starting EXR conversion...")
            print(f"{'=' * 60}")
            try:
                import depth_convert
                depth_convert.convert_exr_files(depth_exr_dir, colormap)
                print(f"{'=' * 60}")
                print("EXR conversion complete!")
                print(f"{'=' * 60}")
            except Exception as e:
                print(f"EXR conversion failed: {e}")

    return all(r["success"] for r in results)


def main():
    parser = argparse.ArgumentParser(
        description="Parallel multi-GPU rendering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parallel render frames 1-240 on 8 GPUs:
  python parallel_render.py input.blend -o scene/ --frame-start 1 --frame-end 240 --num-gpus 8

  # Use GPU indices 3,4,5,6,7:
  python parallel_render.py input.blend -o scene/ --frame-start 1 --frame-end 240 --gpu-ids 3,4,5,6,7

  # Use environment variable to set GPUs:
  CUDA_VISIBLE_DEVICES=3,4,5,6,7 python parallel_render.py input.blend -o scene/ --frame-start 1 --frame-end 240

  # Use OPTIX on 4 GPUs:
  python parallel_render.py input.blend -o scene/ --frame-start 1 --frame-end 100 --num-gpus 4 --compute-type OPTIX
        """,
    )

    parser.add_argument("blend_file", help="Path to input .blend file")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--frame-start", type=int, required=True, help="Start frame")
    parser.add_argument("--frame-end", type=int, required=True, help="End frame")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs to use (if --gpu-ids is not set; default: auto-detect or 8)")
    parser.add_argument("--gpu-ids",
                        help="GPU indices to use, e.g. '3,4,5,6,7' (or use CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--frame-step", type=int, default=1, help="Frame step (default: 1)")
    parser.add_argument("--compute-type", default="CUDA",
                        choices=["CUDA", "OPTIX", "HIP", "METAL", "ONEAPI"],
                        help="GPU compute backend (default: CUDA)")
    parser.add_argument("-c", "--camera", help="Camera name")
    parser.add_argument("-w", "--width", type=int, help="Render width")
    parser.add_argument("--height", type=int, help="Render height")
    parser.add_argument("--skip-conversion", action="store_true", help="Skip EXR conversion")
    parser.add_argument("--colormap", default="turbo", help="PNG colormap")
    parser.add_argument("--blender", help="Path to Blender executable")
    parser.add_argument("--no-compositor", action="store_true",
                        help="Auto-create compositor nodes instead of requiring preconfigured compositor")

    args = parser.parse_args()
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
    num_gpus = args.num_gpus if args.num_gpus is not None else (len(gpu_ids) if gpu_ids else 8)

    success = parallel_render(
        args.blend_file,
        args.output,
        args.frame_start,
        args.frame_end,
        num_gpus=num_gpus,
        gpu_ids=gpu_ids,
        frame_step=args.frame_step,
        compute_type=args.compute_type,
        camera=args.camera,
        width=args.width,
        height=args.height,
        skip_conversion=args.skip_conversion,
        colormap=args.colormap,
        blender_exe=args.blender,
        use_compositor=not args.no_compositor,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
