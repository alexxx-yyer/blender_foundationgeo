#!/usr/bin/env python3
"""渲染 + 相机导出 + 转换的组合管线"""

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
                      frame_step: int = 1):
    """在 Blender 中渲染 RGB 和 Depth，并导出相机参数"""
    if not IN_BLENDER:
        raise RuntimeError("此函数必须在 Blender 环境中运行")

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
    )

    focal_dir = os.path.join(output_dir, "focal")
    pose_dir = os.path.join(output_dir, "pose")
    print("  输出目录:")
    print(f"    - Focal: {focal_dir}")
    print(f"    - Pose: {pose_dir}")

    return result


def find_blender_executable():
    """查找 Blender 可执行文件"""
    possible_paths = [
        # 用户目录下的常见安装位置
        os.path.expanduser("~/blender-4.2.17-linux-x64/blender"),
        os.path.expanduser("~/blender-4.2.0-linux-x64/blender"),
        os.path.expanduser("~/blender-3.6.5-linux-x64/blender"),
        # 系统路径
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
        raise RuntimeError("EXR 转换只能在外部环境中运行")
    return depth_convert.convert_single_exr(exr_file, depth_exr_dir, colormap, silent)


def convert_exr_files(depth_exr_dir: str, colormap: str = "turbo"):
    if IN_BLENDER:
        raise RuntimeError("EXR 转换只能在外部环境中运行")
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
                  verbose: bool = False):
    """
    外部主函数：调用 Blender 进行渲染，然后执行转换
    """
    blend_file = os.path.expanduser(blend_file)
    output_dir = os.path.expanduser(output_dir)

    if not os.path.exists(blend_file):
        raise FileNotFoundError(f"找不到文件: {blend_file}")

    if blender_exe is None:
        blender_exe = find_blender_executable()
        if blender_exe is None:
            raise RuntimeError("找不到 Blender 可执行文件，请使用 --blender 参数指定路径")

    if verbose:
        print(f"使用 Blender: {blender_exe}")

    script_path = os.path.join(os.path.dirname(__file__), "render_and_convert.py")

    # 通过环境变量传递 verbose 标志
    env = os.environ.copy()
    env["FG_VERBOSE"] = "1" if verbose else "0"

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

    if verbose:
        print("\n开始 Blender 渲染...\n")
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
                # 过滤 Blender 的详细渲染输出（以 "Fra:" 开头的行）
                # 但保留错误、警告和重要信息
                is_render_progress = line.startswith("Fra:")
                is_error = any(keyword in line for keyword in ["Error", "错误", "Warning", "警告", "Traceback", "Exception"])
                is_important = any(keyword in line for keyword in ["Saved:", "渲染进度", "完成", "失败"])
                
                # 显示条件：
                # 1. verbose 模式下显示所有内容
                # 2. 非 verbose 模式下只显示错误/警告/重要信息，不显示渲染进度
                should_print = verbose or (not is_render_progress and (is_error or is_important))
                
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
                                # 转换错误始终显示
                                print(f"  警告: 转换失败 {os.path.basename(exr_file)}: {e}",
                                      file=sys.stderr)

    returncode = process.wait()

    if returncode != 0:
        print(f"\nBlender 渲染失败 (退出码: {returncode})")
        return False

    if not skip_conversion:
        if verbose:
            print("\n检查是否有遗漏的 EXR 文件...")
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
                        print(f"  警告: 转换失败 {os.path.basename(exr_file)}: {e}",
                              file=sys.stderr)

        if verbose:
            if remaining_count > 0:
                print(f"  转换了 {remaining_count} 个遗漏的文件")
            print(f"  总共转换了 {len(converted_files)} 个文件")

    return True
