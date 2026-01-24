#!/usr/bin/env python3
"""整合 Blender 渲染、相机参数导出和 EXR 转换功能"""

import os
import sys
import argparse
import subprocess
import glob
import time
import numpy as np

# 尝试导入 bpy（在 Blender 环境中）
try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

# 导入转换函数（在外部环境中）
if not IN_BLENDER:
    # 添加脚本目录到路径以便导入
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    
    try:
        from exr2npy import exr_to_npy
        from exr2png import exr_to_png
    except ImportError as e:
        print(f"警告: 无法导入转换函数: {e}")
        exr_to_npy = None
        exr_to_png = None


# ==================== Blender 内部函数 ====================

def get_render_device_info():
    """获取渲染设备信息"""
    if not IN_BLENDER:
        return "N/A"

    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons.get('cycles')

    # 获取渲染引擎
    engine = bpy.context.scene.render.engine

    info = {
        'engine': engine,
        'device': 'CPU',
        'gpu_devices': []
    }

    if engine == 'CYCLES' and cycles_prefs:
        cprefs = cycles_prefs.preferences
        info['device'] = bpy.context.scene.cycles.device

        # 获取可用的 GPU 设备
        try:
            # 获取计算设备类型
            compute_device_type = cprefs.compute_device_type
            info['compute_type'] = compute_device_type

            # 获取设备列表
            if hasattr(cprefs, 'get_devices'):
                cprefs.get_devices()

            # 只获取当前计算类型的设备，避免重复
            seen_devices = set()
            if hasattr(cprefs, 'devices'):
                for device in cprefs.devices:
                    if device.use and device.type != 'CPU':
                        # 只显示与当前计算类型匹配的设备，或者只显示设备名（去重）
                        if device.type == compute_device_type:
                            if device.name not in seen_devices:
                                info['gpu_devices'].append(device.name)
                                seen_devices.add(device.name)
        except Exception:
            pass

    elif engine == 'BLENDER_EEVEE_NEXT' or engine == 'BLENDER_EEVEE':
        info['device'] = 'GPU'

    return info


def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {mins}m {secs:.0f}s"


def print_progress_bar(current, total, frame_time=None, elapsed=None, prefix='渲染进度'):
    """打印进度条"""
    bar_length = 30
    progress = current / total if total > 0 else 0
    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)
    percent = progress * 100

    # 估算剩余时间
    eta_str = ""
    if elapsed and current > 0:
        eta = (elapsed / current) * (total - current)
        eta_str = f" | ETA: {format_time(eta)}"

    frame_str = ""
    if frame_time:
        frame_str = f" | {format_time(frame_time)}/帧"

    # 在 Blender 环境中，\r 可能不工作，所以每 5 帧或最后一帧打印一次
    if current == total or current % 5 == 0 or total <= 5:
        print(f"{prefix}: |{bar}| {percent:.1f}% ({current}/{total}){frame_str}{eta_str}")
        sys.stdout.flush()


def get_camera_intrinsics(camera, render_width, render_height):
    """
    计算相机内参（焦距）
    
    Args:
        camera: Blender 相机数据对象
        render_width: 渲染宽度
        render_height: 渲染高度
    
    Returns:
        focal_length_px, focal_length_py: 焦距（像素单位）
    """
    # 获取传感器尺寸（毫米）
    sensor_width = camera.sensor_width
    sensor_height = camera.sensor_height
    
    # 获取焦距（毫米）
    focal_length_mm = camera.lens
    
    # 计算焦距（像素单位）
    focal_length_px = (focal_length_mm / sensor_width) * render_width
    focal_length_py = (focal_length_mm / sensor_height) * render_height
    
    return focal_length_px, focal_length_py


def get_camera_pose(camera_obj, use_evaluated=True):
    """
    获取相机位姿（4x4 变换矩阵，从世界坐标到相机坐标）
    
    Args:
        camera_obj: Blender 相机对象
        use_evaluated: 是否使用评估后的对象（用于动画）
    
    Returns:
        pose_matrix: 4x4 numpy 数组，表示从世界坐标到相机坐标的变换
    """
    if use_evaluated:
        try:
            if hasattr(bpy.context, 'evaluated_depsgraph_get'):
                depsgraph = bpy.context.evaluated_depsgraph_get()
                camera_eval = camera_obj.evaluated_get(depsgraph)
                matrix_world = camera_eval.matrix_world
            elif hasattr(bpy.context, 'depsgraph'):
                depsgraph = bpy.context.depsgraph
                camera_eval = camera_obj.evaluated_get(depsgraph)
                matrix_world = camera_eval.matrix_world
            else:
                matrix_world = camera_obj.matrix_world
        except:
            matrix_world = camera_obj.matrix_world
    else:
        matrix_world = camera_obj.matrix_world
    
    # 转换为 numpy 数组
    pose = np.array(matrix_world)
    
    # 坐标系转换（从 Blender 坐标系到 CV 坐标系）
    coord_transform = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    
    pose_cv = coord_transform @ pose
    return pose_cv


def render_and_export(blend_path: str, output_dir: str,
                      camera_name: str | None = None,
                      render_width: int | None = None,
                      render_height: int | None = None,
                      export_animation: bool = False,
                      frame_start: int | None = None,
                      frame_end: int | None = None,
                      frame_step: int = 1):
    """
    在 Blender 中渲染 RGB 和 Depth，并导出相机参数
    
    Args:
        blend_path: .blend 文件路径
        output_dir: 输出目录（scene/）
        camera_name: 相机名称（默认：活动相机）
        render_width: 渲染宽度
        render_height: 渲染高度
        export_animation: 是否导出动画
        frame_start: 起始帧
        frame_end: 结束帧
        frame_step: 帧步长
    """
    if not IN_BLENDER:
        raise RuntimeError("此函数必须在 Blender 环境中运行")

    blend_path = os.path.expanduser(blend_path)
    output_dir = os.path.abspath(os.path.expanduser(output_dir))

    # 先显示基本信息
    print(f"\n{'='*60}")
    print(f"渲染任务")
    print(f"{'='*60}")
    print(f"  Blender 版本: {bpy.app.version_string}")
    sys.stdout.flush()

    # 打开 Blender 文件
    print(f"  加载文件: {blend_path}")
    sys.stdout.flush()
    bpy.ops.wm.open_mainfile(filepath=blend_path)
    
    # 获取场景
    scene = bpy.context.scene
    view_layer = scene.view_layers[0]
    
    # 获取渲染尺寸
    if render_width is None:
        render_width = scene.render.resolution_x
    if render_height is None:
        render_height = scene.render.resolution_y
    
    # 获取相机
    all_cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    
    if camera_name is None:
        camera_obj = scene.camera
        if camera_obj is None:
            if all_cameras:
                raise ValueError("场景中没有活动相机，请使用 -c 参数指定相机名称")
            else:
                raise ValueError("场景中没有相机对象")
    else:
        camera_obj = bpy.data.objects.get(camera_name)
        if camera_obj is None or camera_obj.type != 'CAMERA':
            camera_name_lower = camera_name.lower()
            for obj in all_cameras:
                if obj.name.lower() == camera_name_lower or camera_name_lower in obj.name.lower():
                    camera_obj = obj
                    print(f"找到匹配的相机: {obj.name} (搜索: {camera_name})")
                    break
        
        if camera_obj is None or camera_obj.type != 'CAMERA':
            raise ValueError(f"找不到名为 '{camera_name}' 的相机对象")
    
    # 设置活动相机
    scene.camera = camera_obj
    camera_data = camera_obj.data
    
    # 创建输出目录
    rgb_dir = os.path.join(output_dir, 'rgb')
    depth_exr_dir = os.path.join(output_dir, 'depth', 'exr')
    focal_dir = os.path.join(output_dir, 'focal')
    pose_dir = os.path.join(output_dir, 'pose')
    
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_exr_dir, exist_ok=True)
    os.makedirs(focal_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)
    
    # 启用深度通道
    view_layer.use_pass_z = True
    
    # 获取帧范围
    if export_animation:
        if frame_start is None:
            frame_start = scene.frame_start
        if frame_end is None:
            frame_end = scene.frame_end
    else:
        frame_start = scene.frame_current
        frame_end = scene.frame_current
    
    # 计算总帧数
    total_frames = len(range(frame_start, frame_end + 1, frame_step))

    # 显示渲染信息（继续）
    print(f"  渲染引擎: {scene.render.engine}")

    # 获取设备信息
    device_info = get_render_device_info()
    if isinstance(device_info, dict):
        print(f"  渲染设备: {device_info['device']}")
        if device_info.get('gpu_devices'):
            for gpu in device_info['gpu_devices']:
                print(f"    - {gpu}")
        if device_info.get('compute_type'):
            print(f"  计算类型: {device_info['compute_type']}")

    print(f"  分辨率: {render_width} x {render_height}")
    print(f"  帧范围: {frame_start} - {frame_end} (步长: {frame_step})")
    print(f"  总帧数: {total_frames}")
    print(f"  输出目录: {output_dir}")
    print(f"{'='*60}")
    print(f"")
    sys.stdout.flush()

    # ========== 查找合成器节点树（只执行一次）==========
    tree = None

    # 方法1: 从 scene.node_tree 获取（Blender 4.x 及更早）
    try:
        if hasattr(scene, 'node_tree') and scene.node_tree:
            tree = scene.node_tree
    except Exception:
        pass

    # 方法2: 从 bpy.data.node_groups 中查找合成器节点树（Blender 5.0+）
    if tree is None:
        try:
            compositor_trees = [ng for ng in bpy.data.node_groups if ng.bl_idname == 'CompositorNodeTree']
            for ng in compositor_trees:
                if 'composit' in ng.name.lower():
                    tree = ng
                    break
            if tree is None and compositor_trees:
                tree = compositor_trees[0]
        except Exception:
            pass

    if tree:
        print(f"  合成器节点: {tree.name} ({len(tree.nodes)} nodes)")
    sys.stdout.flush()

    # 辅助函数：追踪节点链
    def trace_node_chain(node, visited=None):
        """递归遍历节点链，找到源头节点类型"""
        if visited is None:
            visited = set()
        if node in visited:
            return None
        visited.add(node)
        if node.type == 'R_LAYERS':
            return 'R_LAYERS'
        for input_slot in node.inputs:
            if input_slot.is_linked:
                for link in input_slot.links:
                    result = trace_node_chain(link.from_node, visited)
                    if result:
                        return result
        return None

    def get_source_output_name(node, link_from_socket=None, visited=None):
        """获取节点链源头的输出名称"""
        if visited is None:
            visited = set()
        if node in visited:
            return None
        visited.add(node)
        if node.type == 'R_LAYERS':
            if link_from_socket:
                return link_from_socket.name.lower()
            for output in node.outputs:
                if output.is_linked:
                    return output.name.lower()
            return None
        for input_slot in node.inputs:
            if input_slot.is_linked:
                for link in input_slot.links:
                    result = get_source_output_name(link.from_node, link.from_socket, visited)
                    if result:
                        return result
        return None

    # 查找 RGB 和 Depth 文件输出节点
    rgb_file_output = None
    depth_file_output = None

    if tree:
        for node in tree.nodes:
            if node.type == 'OUTPUT_FILE':
                for input_slot in node.inputs:
                    if input_slot.is_linked:
                        link = input_slot.links[0]
                        source_output_name = get_source_output_name(link.from_node, link.from_socket)
                        if not source_output_name:
                            source_output_name = link.from_socket.name.lower()
                        if 'depth' in source_output_name or 'z' in source_output_name or 'v' in source_output_name:
                            depth_file_output = node
                            break
                        elif 'image' in source_output_name or 'rgba' in source_output_name or 'rgb' in source_output_name:
                            rgb_file_output = node
                            break
                if not any(slot.is_linked for slot in node.inputs):
                    if rgb_file_output is None:
                        rgb_file_output = node
                    elif depth_file_output is None:
                        depth_file_output = node

        if not rgb_file_output:
            raise RuntimeError("错误: 在合成器中没有找到 RGB 文件输出节点！")
        if not depth_file_output:
            raise RuntimeError("错误: 在合成器中没有找到 Depth 文件输出节点！")
        print("✓ 已找到 RGB 和 Depth 文件输出节点")
        sys.stdout.flush()
    else:
        raise RuntimeError("错误: 无法访问合成器节点树！")

    # 辅助函数：设置 File Output 节点的路径
    def set_file_output_path(node, directory, filename):
        if hasattr(node, 'base_path'):
            node.base_path = directory + os.sep
        elif hasattr(node, 'directory'):
            node.directory = directory + os.sep
        if hasattr(node, 'file_slots') and len(node.file_slots) > 0:
            node.file_slots[0].path = filename
        elif hasattr(node, 'file_output_items') and len(node.file_output_items) > 0:
            if hasattr(node, 'file_name'):
                node.file_name = filename
            item = node.file_output_items[0]
            if hasattr(item, 'name'):
                item.name = filename
        elif hasattr(node, 'file_name'):
            node.file_name = filename

    def set_exr_format(format_obj):
        try:
            format_obj.file_format = 'OPEN_EXR'
        except TypeError:
            format_obj.file_format = 'OPEN_EXR_MULTILAYER'

    # ========== 帧循环 ==========
    frames_rendered = 0
    render_start_time = time.time()
    frame_times = []

    for frame in range(frame_start, frame_end + 1, frame_step):
        frame_start_time = time.time()
        frame_str = f"{frame:06d}"

        # 设置当前帧
        scene.frame_set(frame)
        bpy.context.view_layer.update()

        # 更新输出路径
        if rgb_file_output:
            set_file_output_path(rgb_file_output, rgb_dir, frame_str)
            rgb_file_output.format.file_format = 'PNG'

        if depth_file_output:
            set_file_output_path(depth_file_output, depth_exr_dir, frame_str)
            set_exr_format(depth_file_output.format)
            depth_file_output.format.color_depth = '32'
            try:
                depth_file_output.format.color_mode = 'BW'
            except (AttributeError, TypeError):
                try:
                    depth_file_output.format.color_mode = 'RGB'
                except:
                    pass

        # 执行渲染
        bpy.ops.render.render(write_still=False)

        # 重命名文件
        rgb_files = glob.glob(os.path.join(rgb_dir, f"{frame_str}*.png"))
        if rgb_files:
            rgb_file = max(rgb_files, key=os.path.getctime)
            target_rgb = os.path.join(rgb_dir, f"{frame_str}.png")
            if rgb_file != target_rgb:
                if os.path.exists(target_rgb):
                    os.remove(target_rgb)
                os.rename(rgb_file, target_rgb)

        depth_files = glob.glob(os.path.join(depth_exr_dir, f"{frame_str}*.exr"))
        if depth_files:
            depth_file = max(depth_files, key=os.path.getctime)
            target_depth = os.path.join(depth_exr_dir, f"{frame_str}.exr")
            if depth_file != target_depth:
                if os.path.exists(target_depth):
                    os.remove(target_depth)
                os.rename(depth_file, target_depth)

        # 导出相机参数
        fx, fy = get_camera_intrinsics(camera_data, render_width, render_height)
        pose = get_camera_pose(camera_obj, use_evaluated=True)

        focal_path = os.path.join(focal_dir, f"{frame_str}.txt")
        with open(focal_path, 'w') as f:
            f.write(f"{fx:.6f}\n" if abs(fx - fy) < 1e-6 else f"{fx:.6f} {fy:.6f}\n")

        pose_path = os.path.join(pose_dir, f"{frame_str}.txt")
        np.savetxt(pose_path, pose, fmt='%.8f')

        frames_rendered += 1
        frame_elapsed = time.time() - frame_start_time
        frame_times.append(frame_elapsed)
        total_elapsed = time.time() - render_start_time
        avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
        print_progress_bar(frames_rendered, total_frames, avg_frame_time, total_elapsed)

    # 渲染完成统计
    total_time = time.time() - render_start_time
    avg_time = total_time / frames_rendered if frames_rendered > 0 else 0

    print(f"\n\n{'='*60}")
    print(f"渲染完成!")
    print(f"{'='*60}")
    print(f"  总帧数: {frames_rendered}")
    print(f"  总用时: {format_time(total_time)}")
    print(f"  平均每帧: {format_time(avg_time)}")
    print(f"  输出目录:")
    print(f"    - RGB: {rgb_dir}")
    print(f"    - Depth EXR: {depth_exr_dir}")
    print(f"    - Focal: {focal_dir}")
    print(f"    - Pose: {pose_dir}")
    print(f"{'='*60}")


# ==================== 外部包装函数 ====================

def find_blender_executable():
    """查找 Blender 可执行文件"""
    possible_paths = [
        'blender',
        '/usr/bin/blender',
        '/usr/local/bin/blender',
        '/opt/blender/blender',
    ]
    
    for path in possible_paths:
        try:
            result = subprocess.run(
                [path, '--version'],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                return path
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    return None


def convert_exr_files(depth_exr_dir: str, colormap: str = 'turbo'):
    """
    将 depth/exr/ 目录中的 EXR 文件转换为 NPY 和 PNG
    
    Args:
        depth_exr_dir: depth/exr/ 目录路径
        colormap: PNG 转换的 colormap
    """
    if exr_to_npy is None or exr_to_png is None:
        print("警告: EXR 转换函数不可用，跳过转换")
        return
    
    depth_exr_dir = os.path.expanduser(depth_exr_dir)
    
    if not os.path.isdir(depth_exr_dir):
        print(f"警告: 目录不存在: {depth_exr_dir}")
        return
    
    # 查找所有 EXR 文件
    exr_files = sorted(glob.glob(os.path.join(depth_exr_dir, "*.exr")))
    
    if not exr_files:
        print(f"在目录 {depth_exr_dir} 中未找到 EXR 文件")
        return
    
    print(f"\n找到 {len(exr_files)} 个 EXR 文件，开始转换...")
    
    # 创建输出目录
    depth_npy_dir = os.path.join(os.path.dirname(depth_exr_dir), 'npy')
    depth_vis_dir = os.path.join(os.path.dirname(depth_exr_dir), 'vis')
    os.makedirs(depth_npy_dir, exist_ok=True)
    os.makedirs(depth_vis_dir, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    
    for exr_file in exr_files:
        try:
            base_name = os.path.basename(exr_file)
            base_name_no_ext = os.path.splitext(base_name)[0]
            
            # 转换为 NPY
            npy_path = os.path.join(depth_npy_dir, f"{base_name_no_ext}.npy")
            exr_to_npy(exr_file, npy_path)
            
            # 转换为 PNG
            png_path = os.path.join(depth_vis_dir, f"{base_name_no_ext}.png")
            exr_to_png(exr_file, png_path, colormap=colormap)
            
            success_count += 1
        except Exception as e:
            print(f"  ✗ 转换失败 {os.path.basename(exr_file)}: {e}")
            fail_count += 1
    
    print(f"\n转换完成!")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  NPY: {depth_npy_dir}")
    print(f"  PNG: {depth_vis_dir}")


def main_external(blend_file: str, output_dir: str,
                  camera_name: str | None = None,
                  render_width: int | None = None,
                  render_height: int | None = None,
                  export_animation: bool = False,
                  frame_start: int | None = None,
                  frame_end: int | None = None,
                  frame_step: int = 1,
                  skip_conversion: bool = False,
                  colormap: str = 'turbo',
                  blender_exe: str | None = None):
    """
    外部主函数：调用 Blender 进行渲染，然后执行转换
    """
    blend_file = os.path.expanduser(blend_file)
    output_dir = os.path.expanduser(output_dir)
    
    if not os.path.exists(blend_file):
        raise FileNotFoundError(f"找不到文件: {blend_file}")
    
    # 查找 Blender 可执行文件
    if blender_exe is None:
        blender_exe = find_blender_executable()
        if blender_exe is None:
            raise RuntimeError("找不到 Blender 可执行文件，请使用 --blender 参数指定路径")
    
    print(f"使用 Blender: {blender_exe}")
    
    # 获取脚本路径
    script_path = os.path.abspath(__file__)
    
    # 构建命令
    cmd = [
        blender_exe,
        '--background',
        '--python', script_path,
        '--',
        blend_file,
        '--output', output_dir
    ]
    
    if camera_name:
        cmd.extend(['--camera', camera_name])
    if render_width:
        cmd.extend(['--width', str(render_width)])
    if render_height:
        cmd.extend(['--height', str(render_height)])
    if export_animation:
        cmd.append('--export-animation')
        if frame_start is not None:
            cmd.extend(['--frame-start', str(frame_start)])
        if frame_end is not None:
            cmd.extend(['--frame-end', str(frame_end)])
        if frame_step != 1:
            cmd.extend(['--frame-step', str(frame_step)])
    
    # 执行 Blender 渲染（实时输出）
    print("\n开始 Blender 渲染...\n")
    sys.stdout.flush()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1  # 行缓冲
    )

    # 实时读取并显示输出
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            # 过滤掉一些不必要的 Blender 输出
            line = line.rstrip()
            if line:
                print(line)
                sys.stdout.flush()

    returncode = process.wait()

    if returncode != 0:
        print(f"\nBlender 渲染失败 (退出码: {returncode})")
        return False
    
    # 执行 EXR 转换
    if not skip_conversion:
        depth_exr_dir = os.path.join(output_dir, 'depth', 'exr')
        convert_exr_files(depth_exr_dir, colormap)
    
    return True


# ==================== 主入口 ====================

if __name__ == "__main__":
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        argv = sys.argv[1:]
    
    parser = argparse.ArgumentParser(
        description="整合 Blender 渲染、相机参数导出和 EXR 转换",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 单帧渲染:
  python render_and_convert.py input.blend -o scene/
  
  # 动画渲染:
  python render_and_convert.py input.blend -o scene/ --export-animation
  
  # 指定相机和渲染尺寸:
  python render_and_convert.py input.blend -o scene/ -c Camera -w 1920 --height 1080
  
  # 指定帧范围:
  python render_and_convert.py input.blend -o scene/ --export-animation --frame-start 1 --frame-end 48
  
  # 跳过 EXR 转换:
  python render_and_convert.py input.blend -o scene/ --skip-conversion
        """
    )
    
    parser.add_argument("blend_file", help="输入的 .blend 文件路径")
    parser.add_argument("-o", "--output", required=True, help="输出目录（scene/）")
    parser.add_argument("-c", "--camera", help="相机名称（默认：活动相机）")
    parser.add_argument("-w", "--width", type=int, help="渲染宽度（默认：使用场景设置）")
    parser.add_argument("--height", type=int, help="渲染高度（默认：使用场景设置）")
    parser.add_argument("--export-animation", action="store_true",
                       help="导出动画中每一帧")
    parser.add_argument("--frame-start", type=int, default=None,
                       help="起始帧（默认：使用场景设置）")
    parser.add_argument("--frame-end", type=int, default=None,
                       help="结束帧（默认：使用场景设置）")
    parser.add_argument("--frame-step", type=int, default=1,
                       help="帧步长（默认：1）")
    parser.add_argument("--skip-conversion", action="store_true",
                       help="跳过 EXR 转换（仅渲染）")
    parser.add_argument("--colormap", default="turbo",
                       help="PNG 转换的 colormap（默认：turbo）")
    parser.add_argument("--blender", help="Blender 可执行文件路径（默认：自动查找）")
    
    args = parser.parse_args(argv)
    
    if IN_BLENDER:
        # 在 Blender 环境中运行
        try:
            render_and_export(
                args.blend_file,
                args.output,
                args.camera,
                args.width,
                args.height,
                args.export_animation,
                args.frame_start,
                args.frame_end,
                args.frame_step
            )
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # 在外部环境中运行
        try:
            main_external(
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
                args.blender
            )
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)
