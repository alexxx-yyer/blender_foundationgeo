#!/usr/bin/env python3
"""多进程并行渲染：每张 GPU 渲染不同的帧范围"""

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
    """从 .blend 文件中读取渲染信息（分辨率、采样数等）"""
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


def render_worker(args: dict) -> dict:
    """单个渲染工作进程"""
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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "render_and_convert.py")

    env = os.environ.copy()
    env["FG_DEVICE"] = "GPU"
    env["FG_COMPUTE_TYPE"] = compute_type
    # 设置 CUDA_VISIBLE_DEVICES 限制进程只能看到指定的 GPU
    # 这样初始化也会在该 GPU 上进行，而不是都在 GPU 0 上
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # 由于 CUDA_VISIBLE_DEVICES 限制后，可见的 GPU 索引变成 0
    env["FG_GPU_IDS"] = "0"

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

    print(f"[GPU {gpu_id}] 开始渲染帧 {frame_start}-{frame_end}")
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
                    # 只显示进度条，过滤掉 Saved: 等其他信息
                    is_progress = line.startswith("渲染进度")
                    is_error = any(kw in line for kw in ["Error", "错误", "Warning", "警告", "Traceback", "Exception"])
                    
                    if is_progress:
                        # 进度条单独一行显示
                        print(f"[GPU {gpu_id}] {line}")
                        sys.stdout.flush()
                    elif is_error:
                        print(f"[GPU {gpu_id}] {line}")
                        sys.stdout.flush()
                        stderr_output.append(line)

        returncode = process.wait()

        if returncode == 0:
            print(f"[GPU {gpu_id}] 完成帧 {frame_start}-{frame_end}")
            return {
                "gpu_id": gpu_id,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "success": True,
            }
        else:
            error_msg = "\n".join(stderr_output[-5:]) if stderr_output else "未知错误"
            print(f"[GPU {gpu_id}] 失败: {error_msg[:200]}")
            return {
                "gpu_id": gpu_id,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "success": False,
                "error": error_msg,
            }
    except Exception as e:
        print(f"[GPU {gpu_id}] 异常: {e}")
        return {
            "gpu_id": gpu_id,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "success": False,
            "error": str(e),
        }


def distribute_frames(frame_start: int, frame_end: int, num_gpus: int, frame_step: int = 1):
    """将帧范围分配给多个 GPU"""
    frames = list(range(frame_start, frame_end + 1, frame_step))
    total_frames = len(frames)
    frames_per_gpu = math.ceil(total_frames / num_gpus)

    distributions = []
    for i in range(num_gpus):
        start_idx = i * frames_per_gpu
        end_idx = min(start_idx + frames_per_gpu, total_frames)

        if start_idx >= total_frames:
            break

        gpu_frames = frames[start_idx:end_idx]
        if gpu_frames:
            distributions.append({
                "gpu_id": i,
                "frame_start": gpu_frames[0],
                "frame_end": gpu_frames[-1],
            })

    return distributions


def parallel_render(
    blend_file: str,
    output_dir: str,
    frame_start: int,
    frame_end: int,
    num_gpus: int = 8,
    frame_step: int = 1,
    compute_type: str = "CUDA",
    camera: str = None,
    width: int = None,
    height: int = None,
    skip_conversion: bool = False,
    colormap: str = "turbo",
    blender_exe: str = None,
):
    """多 GPU 并行渲染"""
    if blender_exe is None:
        blender_exe = find_blender_executable()
        if blender_exe is None:
            raise RuntimeError("找不到 Blender，请使用 --blender 参数指定路径")
    else:
        blender_exe = os.path.expanduser(blender_exe)

    blend_file = os.path.expanduser(blend_file)
    output_dir = os.path.expanduser(output_dir)

    if not os.path.exists(blend_file):
        raise FileNotFoundError(f"找不到文件: {blend_file}")

    os.makedirs(output_dir, exist_ok=True)

    # 获取 .blend 文件信息
    print("正在读取场景信息...")
    blend_info = get_blend_info(blend_file, blender_exe)
    
    # 使用命令行参数覆盖或使用 .blend 文件中的值
    render_width = width if width else blend_info.get("width", "未知")
    render_height = height if height else blend_info.get("height", "未知")
    samples = blend_info.get("samples", "未知")
    engine = blend_info.get("engine", "未知")

    distributions = distribute_frames(frame_start, frame_end, num_gpus, frame_step)

    print(f"\n{'=' * 60}")
    print("多 GPU 并行渲染")
    print(f"{'=' * 60}")
    print(f"  Blender: {blender_exe}")
    print(f"  输入文件: {blend_file}")
    print(f"  输出目录: {output_dir}")
    print(f"  渲染引擎: {engine}")
    print(f"  分辨率: {render_width} x {render_height}")
    print(f"  采样数: {samples}")
    print(f"  帧范围: {frame_start} - {frame_end} (步长: {frame_step})")
    print(f"  GPU 数量: {num_gpus}")
    print(f"  计算类型: {compute_type}")
    print(f"\n帧分配:")
    for dist in distributions:
        frames_count = len(range(dist["frame_start"], dist["frame_end"] + 1, frame_step))
        print(f"  GPU {dist['gpu_id']}: 帧 {dist['frame_start']}-{dist['frame_end']} ({frames_count} 帧)")
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
        })

    results = []
    with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        futures = {executor.submit(render_worker, task): task for task in tasks}

        # 等待所有任务完成，子进程会实时输出进度
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count

    print(f"\n{'=' * 60}")
    print("渲染完成!")
    print(f"{'=' * 60}")
    print(f"  成功: {success_count}/{len(results)}")
    if fail_count > 0:
        print(f"  失败: {fail_count}")
        for r in results:
            if not r["success"]:
                print(f"    - GPU {r['gpu_id']}: 帧 {r['frame_start']}-{r['frame_end']}")
    print(f"{'=' * 60}")

    # EXR 转换
    if not skip_conversion and success_count > 0:
        depth_exr_dir = os.path.join(output_dir, "depth", "exr")
        if os.path.isdir(depth_exr_dir):
            print(f"\n{'=' * 60}")
            print("开始 EXR 转换...")
            print(f"{'=' * 60}")
            try:
                import depth_convert
                depth_convert.convert_exr_files(depth_exr_dir, colormap)
                print(f"{'=' * 60}")
                print("EXR 转换完成!")
                print(f"{'=' * 60}")
            except Exception as e:
                print(f"EXR 转换失败: {e}")

    return all(r["success"] for r in results)


def main():
    parser = argparse.ArgumentParser(
        description="多 GPU 并行渲染",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 8 卡并行渲染 1-240 帧:
  python parallel_render.py input.blend -o scene/ --frame-start 1 --frame-end 240 --num-gpus 8

  # 4 卡并行，指定 OPTIX:
  python parallel_render.py input.blend -o scene/ --frame-start 1 --frame-end 100 --num-gpus 4 --compute-type OPTIX
        """,
    )

    parser.add_argument("blend_file", help="输入的 .blend 文件路径")
    parser.add_argument("-o", "--output", required=True, help="输出目录")
    parser.add_argument("--frame-start", type=int, required=True, help="起始帧")
    parser.add_argument("--frame-end", type=int, required=True, help="结束帧")
    parser.add_argument("--num-gpus", type=int, default=8, help="使用的 GPU 数量（默认：8）")
    parser.add_argument("--frame-step", type=int, default=1, help="帧步长（默认：1）")
    parser.add_argument("--compute-type", default="CUDA",
                        choices=["CUDA", "OPTIX", "HIP", "METAL", "ONEAPI"],
                        help="GPU 计算类型（默认：CUDA）")
    parser.add_argument("-c", "--camera", help="相机名称")
    parser.add_argument("-w", "--width", type=int, help="渲染宽度")
    parser.add_argument("--height", type=int, help="渲染高度")
    parser.add_argument("--skip-conversion", action="store_true", help="跳过 EXR 转换")
    parser.add_argument("--colormap", default="turbo", help="PNG colormap")
    parser.add_argument("--blender", help="Blender 可执行文件路径")

    args = parser.parse_args()

    success = parallel_render(
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

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
