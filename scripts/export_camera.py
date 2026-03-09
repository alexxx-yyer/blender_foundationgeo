#!/usr/bin/env python3
"""Export camera focal length and pose (focal / pose)."""

import argparse
import glob
import os
import subprocess
import sys

import numpy as np


try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


def get_camera_intrinsics(camera, render_width, render_height):
    """
    Compute camera intrinsics (focal lengths in pixels).
    """
    sensor_width = camera.sensor_width
    sensor_height = camera.sensor_height
    focal_length_mm = camera.lens

    focal_length_px = (focal_length_mm / sensor_width) * render_width
    focal_length_py = (focal_length_mm / sensor_height) * render_height

    return focal_length_px, focal_length_py


def get_camera_pose(camera_obj, use_evaluated=True):
    """
    Get camera pose as a 4x4 matrix (world-to-camera convention in this pipeline).
    """
    if use_evaluated:
        try:
            if hasattr(bpy.context, "evaluated_depsgraph_get"):
                depsgraph = bpy.context.evaluated_depsgraph_get()
                camera_eval = camera_obj.evaluated_get(depsgraph)
                matrix_world = camera_eval.matrix_world
            elif hasattr(bpy.context, "depsgraph"):
                depsgraph = bpy.context.depsgraph
                camera_eval = camera_obj.evaluated_get(depsgraph)
                matrix_world = camera_eval.matrix_world
            else:
                matrix_world = camera_obj.matrix_world
        except Exception:
            matrix_world = camera_obj.matrix_world
    else:
        matrix_world = camera_obj.matrix_world

    pose = np.array(matrix_world)

    coord_transform = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ])

    pose_cv = coord_transform @ pose
    return pose_cv


def export_frame(camera_obj, camera_data, render_width, render_height,
                 output_dir, frame, use_evaluated=True):
    """
    Export focal and pose for one frame into output_dir/focal and output_dir/pose.
    """
    focal_dir = os.path.join(output_dir, "focal")
    pose_dir = os.path.join(output_dir, "pose")
    os.makedirs(focal_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    fx, fy = get_camera_intrinsics(camera_data, render_width, render_height)
    pose = get_camera_pose(camera_obj, use_evaluated=use_evaluated)

    frame_str = f"{frame:06d}"

    focal_path = os.path.join(focal_dir, f"{frame_str}.txt")
    with open(focal_path, "w") as f:
        f.write(f"{fx:.6f}\n" if abs(fx - fy) < 1e-6 else f"{fx:.6f} {fy:.6f}\n")

    pose_path = os.path.join(pose_dir, f"{frame_str}.txt")
    np.savetxt(pose_path, pose, fmt="%.8f")


def export_camera_animation(blend_path: str, output_dir: str,
                            camera_name: str | None = None,
                            render_width: int | None = None,
                            render_height: int | None = None,
                            export_animation: bool = False,
                            frame_start: int | None = None,
                            frame_end: int | None = None,
                            frame_step: int = 1):
    """
    Export focal/pose from a Blender file (single frame or animation).
    """
    if not IN_BLENDER:
        raise RuntimeError("This function must run inside Blender.")

    blend_path = os.path.expanduser(blend_path)
    output_dir = os.path.abspath(os.path.expanduser(output_dir))

    if not os.path.exists(blend_path):
        raise FileNotFoundError(f"File not found: {blend_path}")

    bpy.ops.wm.open_mainfile(filepath=blend_path)
    scene = bpy.context.scene

    if render_width is None:
        render_width = scene.render.resolution_x
    if render_height is None:
        render_height = scene.render.resolution_y

    all_cameras = [obj for obj in bpy.data.objects if obj.type == "CAMERA"]
    if camera_name is None:
        camera_obj = scene.camera
        if camera_obj is None:
            if all_cameras:
                raise ValueError("No active camera in scene. Use -c to specify a camera.")
            raise ValueError("No camera objects found in scene.")
    else:
        camera_obj = bpy.data.objects.get(camera_name)
        if camera_obj is None or camera_obj.type != "CAMERA":
            camera_name_lower = camera_name.lower()
            for obj in all_cameras:
                if obj.name.lower() == camera_name_lower or camera_name_lower in obj.name.lower():
                    camera_obj = obj
                    print(f"Matched camera: {obj.name} (search: {camera_name})")
                    break
        if camera_obj is None or camera_obj.type != "CAMERA":
            raise ValueError(f"Camera object named '{camera_name}' not found.")

    camera_data = camera_obj.data

    if export_animation:
        if frame_start is None:
            frame_start = scene.frame_start
        if frame_end is None:
            frame_end = scene.frame_end
    else:
        frame_start = scene.frame_current
        frame_end = scene.frame_current

    total_frames = len(range(frame_start, frame_end + 1, frame_step))
    print(f"Exporting focal/pose frame range: {frame_start} - {frame_end} (step: {frame_step})")
    print(f"Total frames: {total_frames}")
    print(f"Output directory: {output_dir}")

    frames_exported = 0
    for frame in range(frame_start, frame_end + 1, frame_step):
        scene.frame_set(frame)
        bpy.context.view_layer.update()

        export_frame(
            camera_obj=camera_obj,
            camera_data=camera_data,
            render_width=render_width,
            render_height=render_height,
            output_dir=output_dir,
            frame=frame,
            use_evaluated=True,
        )

        frames_exported += 1
        if frames_exported % 10 == 0 or frame == frame_end:
            print(f"  Exported frame {frame}/{frame_end} ({frames_exported} frame(s))")

    print("\nExport complete!")
    print(f"  Total frames: {frames_exported}")
    print(f"  focal dir: {os.path.join(output_dir, 'focal')}")
    print(f"  pose dir: {os.path.join(output_dir, 'pose')}")


def find_blender_executable():
    """Find Blender executable."""
    possible_paths = [
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


def batch_export_cameras(blend_dir: str, output_base_dir: str | None = None,
                         camera_name: str | None = None,
                         render_width: int | None = None,
                         render_height: int | None = None,
                         export_animation: bool = False,
                         frame_start: int | None = None,
                         frame_end: int | None = None,
                         frame_step: int = 1):
    """
    Batch-process all .blend files in a directory.
    """
    blend_dir = os.path.expanduser(blend_dir)

    if not os.path.isdir(blend_dir):
        raise NotADirectoryError(f"Directory does not exist: {blend_dir}")

    blend_files = glob.glob(os.path.join(blend_dir, "*.blend"))
    blend_files = [f for f in blend_files if not f.endswith(".blend1")]

    if not blend_files:
        print(f"No .blend files found in directory: {blend_dir}")
        return

    print(f"Found {len(blend_files)} .blend file(s)")

    blender_exe = find_blender_executable()
    if blender_exe is None:
        print("Error: Blender executable not found")
        print("Install Blender and ensure it is in PATH, or pass --blender explicitly")
        return

    print(f"Using Blender: {blender_exe}")

    script_path = os.path.abspath(__file__)

    success_count = 0
    fail_count = 0

    for blend_file in sorted(blend_files):
        print(f"\nProcessing: {os.path.basename(blend_file)}")

        if output_base_dir:
            base_name = os.path.splitext(os.path.basename(blend_file))[0]
            output_dir = os.path.join(output_base_dir, base_name)
        else:
            output_dir = os.path.dirname(blend_file)

        cmd = [
            blender_exe,
            "--background",
            "--python", script_path,
            "--",
            blend_file,
            "-o", output_dir,
        ]

        if camera_name:
            cmd.extend(["-c", camera_name])
        if render_width:
            cmd.extend(["-w", str(render_width)])
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

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                print("  ✓ Success")
                if result.stdout:
                    print(f"  {result.stdout.strip()}")
                success_count += 1
            else:
                print(f"  ✗ Failed (exit code: {result.returncode})")
                if result.stderr:
                    print(f"  Error: {result.stderr.strip()}")
                fail_count += 1

        except subprocess.TimeoutExpired:
            print("  ✗ Timeout")
            fail_count += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            fail_count += 1

    print("\nDone!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")


if __name__ == "__main__":
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = sys.argv[1:]

    if argv and argv[0] == "batch":
        parser = argparse.ArgumentParser(
            description="Batch export camera parameters from Blender .blend files",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument("blend_dir", help="Directory containing .blend files")
        parser.add_argument("-o", "--output", help="Base output directory (default: same directory as each .blend file)")
        parser.add_argument("-c", "--camera", help="Camera name (default: active camera)")
        parser.add_argument("-w", "--width", type=int, help="Render width (default: scene setting)")
        parser.add_argument("--height", type=int, help="Render height (default: scene setting)")
        parser.add_argument("--export-animation", action="store_true",
                            help="Export camera parameters for every animation frame")
        parser.add_argument("--frame-start", type=int, default=None,
                            help="Start frame (default: scene setting)")
        parser.add_argument("--frame-end", type=int, default=None,
                            help="End frame (default: scene setting)")
        parser.add_argument("--frame-step", type=int, default=1,
                            help="Frame step (default: 1)")

        args = parser.parse_args(argv[1:])
        try:
            batch_export_cameras(
                args.blend_dir,
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
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser = argparse.ArgumentParser(
            description="Export focal / pose from a Blender .blend file",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        parser.add_argument("blend_file", help="Path to input .blend file")
        parser.add_argument("-o", "--output", required=True, help="Output directory (scene/)")
        parser.add_argument("-c", "--camera", help="Camera name (default: active camera)")
        parser.add_argument("-w", "--width", type=int, help="Render width (default: scene setting)")
        parser.add_argument("--height", type=int, help="Render height (default: scene setting)")
        parser.add_argument("--export-animation", action="store_true", help="Export every animation frame")
        parser.add_argument("--frame-start", type=int, default=None, help="Start frame (default: scene setting)")
        parser.add_argument("--frame-end", type=int, default=None, help="End frame (default: scene setting)")
        parser.add_argument("--frame-step", type=int, default=1, help="Frame step (default: 1)")

        args = parser.parse_args(argv)

        try:
            export_camera_animation(
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
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
