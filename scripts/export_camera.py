#!/usr/bin/env python3
"""导出相机焦距与位姿（focal / pose）"""

import argparse
import os
import sys

import numpy as np


try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


def get_camera_intrinsics(camera, render_width, render_height):
    """
    计算相机内参（焦距）
    """
    sensor_width = camera.sensor_width
    sensor_height = camera.sensor_height
    focal_length_mm = camera.lens

    focal_length_px = (focal_length_mm / sensor_width) * render_width
    focal_length_py = (focal_length_mm / sensor_height) * render_height

    return focal_length_px, focal_length_py


def get_camera_pose(camera_obj, use_evaluated=True):
    """
    获取相机位姿（4x4 变换矩阵，从世界坐标到相机坐标）
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
    导出指定帧的 focal 与 pose，输出到 output_dir/focal 与 output_dir/pose
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
    从 Blender 文件中导出 focal / pose（单帧或动画）
    """
    if not IN_BLENDER:
        raise RuntimeError("此函数必须在 Blender 环境中运行")

    blend_path = os.path.expanduser(blend_path)
    output_dir = os.path.abspath(os.path.expanduser(output_dir))

    if not os.path.exists(blend_path):
        raise FileNotFoundError(f"找不到文件: {blend_path}")

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
                raise ValueError("场景中没有活动相机，请使用 -c 参数指定相机名称")
            raise ValueError("场景中没有相机对象")
    else:
        camera_obj = bpy.data.objects.get(camera_name)
        if camera_obj is None or camera_obj.type != "CAMERA":
            camera_name_lower = camera_name.lower()
            for obj in all_cameras:
                if obj.name.lower() == camera_name_lower or camera_name_lower in obj.name.lower():
                    camera_obj = obj
                    print(f"找到匹配的相机: {obj.name} (搜索: {camera_name})")
                    break
        if camera_obj is None or camera_obj.type != "CAMERA":
            raise ValueError(f"找不到名为 '{camera_name}' 的相机对象")

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
    print(f"导出 focal/pose 帧范围: {frame_start} - {frame_end} (步长: {frame_step})")
    print(f"总帧数: {total_frames}")
    print(f"输出目录: {output_dir}")

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
            print(f"  已导出帧 {frame}/{frame_end} ({frames_exported} 帧)")

    print("\n导出完成!")
    print(f"  总帧数: {frames_exported}")
    print(f"  focal 目录: {os.path.join(output_dir, 'focal')}")
    print(f"  pose 目录: {os.path.join(output_dir, 'pose')}")


if __name__ == "__main__":
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="从 Blender .blend 文件中导出 focal / pose",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
