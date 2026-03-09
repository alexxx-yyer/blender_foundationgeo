#!/usr/bin/env python3
""" + Cameraexport + """

import glob
import os
import re
import subprocess
import sys
import time


try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

if IN_BLENDER:
    import export_camera
    import render
else:
    import depth_convert


def render_and_export(blend_path: str, output_dir: str,
                      camera_name: str | None = None,
                      render_width: int | None = None,
                      render_height: int | None = None,
                      export_animation: bool = False,
                      frame_start: int | None = None,
                      frame_end: int | None = None,
                      frame_step: int = 1,
                      use_compositor: bool = True):
    """ Blender  RGB  Depth，exportCamera"""
    if not IN_BLENDER:
        raise RuntimeError(" Blender ")

    def _on_frame_rendered(frame, scene, camera_obj, camera_data,
                           render_width, render_height, output_dir):
        export_camera.export_frame(
            camera_obj=camera_obj,
            camera_data=camera_data,
            render_width=render_width,
            render_height=render_height,
            output_dir=output_dir,
            frame=frame,
            use_evaluated=True,
        )

    result = render.render_frames(
        blend_path,
        output_dir,
        camera_name,
        render_width,
        render_height,
        export_animation,
        frame_start,
        frame_end,
        frame_step,
        on_frame_rendered=_on_frame_rendered,
        use_compositor=use_compositor,
    )

    focal_dir = os.path.join(output_dir, "focal")
    pose_dir = os.path.join(output_dir, "pose")
    print("  Output directory:")
    print(f"    - Focal: {focal_dir}")
    print(f"    - Pose: {pose_dir}")

    return result


def find_blender_executable():
    """ Blender executable"""
    possible_paths = [
        # 
        os.path.expanduser("~/blender-4.2.17-linux-x64/blender"),
        os.path.expanduser("~/blender-4.2.0-linux-x64/blender"),
        os.path.expanduser("~/blender-3.6.5-linux-x64/blender"),
        # path
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


def convert_single_exr(exr_file: str, depth_exr_dir: str, colormap: str = "turbo",
                       silent: bool = True):
    if IN_BLENDER:
        raise RuntimeError("EXR ")
    return depth_convert.convert_single_exr(exr_file, depth_exr_dir, colormap, silent)


def convert_exr_files(depth_exr_dir: str, colormap: str = "turbo"):
    if IN_BLENDER:
        raise RuntimeError("EXR ")
    return depth_convert.convert_exr_files(depth_exr_dir, colormap)


def main_external(blend_file: str, output_dir: str,
                  camera_name: str | None = None,
                  render_width: int | None = None,
                  render_height: int | None = None,
                  export_animation: bool = False,
                  frame_start: int | None = None,
                  frame_end: int | None = None,
                  frame_step: int = 1,
                  skip_conversion: bool = False,
                  colormap: str = "turbo",
                  blender_exe: str | None = None,
                  verbose: bool = False,
                  use_compositor: bool = True):
    """
    ： Blender ，
    """
    blend_file = os.path.expanduser(blend_file)
    output_dir = os.path.expanduser(output_dir)

    if not os.path.exists(blend_file):
        raise FileNotFoundError(f"File not found: {blend_file}")

    if blender_exe is None:
        blender_exe = find_blender_executable()
        if blender_exe is None:
            raise RuntimeError("Blender not found executable， --blender path")

    # Compute type
    device = os.environ.get("FG_DEVICE", "CPU")
    compute_type = os.environ.get("FG_COMPUTE_TYPE", "NONE")
    gpu_ids = os.environ.get("FG_GPU_IDS", "")

    # 
    print("\n" + "=" * 50)
    print("")
    print("=" * 50)
    print(f"  Blender:      {blender_exe}")
    print(f"  file:     {blend_file}")
    print(f"  Output directory:     {output_dir}")
    print(f"  :         {device}")
    print(f"  Compute type:     {compute_type}")
    if gpu_ids:
        print(f"  GPU IDs:      {gpu_ids}")
    if camera_name:
        print(f"  Camera:         {camera_name}")
    if render_width or render_height:
        w = render_width if render_width else "default"
        h = render_height if render_height else "default"
        print(f"  Resolution:       {w} x {h}")
    if export_animation:
        start = frame_start if frame_start is not None else "Scenedefault"
        end = frame_end if frame_end is not None else "Scenedefault"
        print(f"  animation:     ")
        print(f"  :       {start} - {end}")
        print(f"  Frame step:       {frame_step}")
    else:
        print(f"  animation:      ()")
    print(f"  :     {'' if skip_conversion else ''}")
    if not skip_conversion:
        print(f"  :   {colormap}")
    print("=" * 50 + "\n")

    script_path = os.path.join(os.path.dirname(__file__), "render_and_convert.py")

    #  verbose  use_compositor 
    env = os.environ.copy()
    env["FG_VERBOSE"] = "1" if verbose else "0"
    env["FG_USE_COMPOSITOR"] = "1" if use_compositor else "0"

    cmd = [
        blender_exe,
        "--background",
        "--python", script_path,
        "--",
        blend_file,
        "--output", output_dir,
    ]

    if camera_name:
        cmd.extend(["--camera", camera_name])
    if render_width:
        cmd.extend(["--width", str(render_width)])
    if render_height:
        cmd.extend(["--height", str(render_height)])
    if export_animation:
        cmd.append("--export-animation")
        if frame_start is not None:
            cmd.extend(["--frame-start", str(frame_start)])
        if frame_end is not None:
            cmd.extend(["--frame-end", str(frame_end)])
        if frame_step != 1:
            cmd.extend(["--frame-step", str(frame_step)])

    print("...")
    sys.stdout.flush()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    depth_exr_dir = os.path.join(output_dir, "depth", "exr")
    converted_files = set()

    exr_pattern = re.compile(r"Saved: '([^']+\\.exr)'")

    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            line = line.rstrip()
            if line:
                #  Blender （ "Fra:" ）
                # Error、Warning
                is_render_progress = line.startswith("Fra:")
                is_saved_message = line.startswith("Saved:")
                is_error = any(keyword in line for keyword in ["Error", "Error", "Warning", "Warning", "Traceback", "Exception"])
                is_important = any(keyword in line for keyword in ["Render progress", "Done", "Failed"])
                
                # ：
                # 1. verbose （ Saved ）
                # 2.  verbose Error/Warning/，Render progress Saved 
                should_print = (verbose or (not is_render_progress and (is_error or is_important))) and not is_saved_message
                
                if should_print:
                    print(line)
                    sys.stdout.flush()

                if not skip_conversion:
                    match = exr_pattern.search(line)
                    if match:
                        exr_file = match.group(1)
                        exr_file = os.path.abspath(exr_file)

                        if exr_file not in converted_files:
                            converted_files.add(exr_file)
                            time.sleep(0.1)

                            try:
                                convert_single_exr(exr_file, depth_exr_dir, colormap, silent=True)
                            except Exception as e:
                                # Error
                                print(f"  Warning: Failed {os.path.basename(exr_file)}: {e}",
                                      file=sys.stderr)

    returncode = process.wait()

    if returncode != 0:
        print(f"\nBlender Failed (: {returncode})")
        return False

    if not skip_conversion:
        if verbose:
            print("\n EXR file...")
        remaining_files = glob.glob(os.path.join(depth_exr_dir, "*.exr"))
        remaining_count = 0
        for exr_file in remaining_files:
            exr_file = os.path.abspath(exr_file)
            if exr_file not in converted_files:
                remaining_count += 1
                try:
                    convert_single_exr(exr_file, depth_exr_dir, colormap, silent=True)
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Failed {os.path.basename(exr_file)}: {e}",
                              file=sys.stderr)

        if verbose:
            if remaining_count > 0:
                print(f"   {remaining_count} file")
            print(f"   {len(converted_files)} file")

    return True
