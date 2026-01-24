#!/usr/bin/env python3
"""Blender 渲染：输出 RGB 与深度 EXR"""

import glob
import os
import sys
import time

try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


def apply_render_device(device: str | None, compute_type: str | None = None) -> None:
    """
    根据配置设置 Blender 渲染设备
    device: CPU / GPU
    compute_type: CUDA / OPTIX / HIP / METAL / ONEAPI
    """
    if not IN_BLENDER or not device:
        return

    device = str(device).strip().upper()
    compute_type = str(compute_type).strip().upper() if compute_type else None

    if compute_type:
        try:
            cycles_prefs = bpy.context.preferences.addons.get("cycles")
            if cycles_prefs:
                cycles_prefs.preferences.compute_device_type = compute_type
                if hasattr(cycles_prefs.preferences, "get_devices"):
                    cycles_prefs.preferences.get_devices()
        except Exception:
            pass

    if device == "CPU":
        bpy.context.scene.cycles.device = "CPU"
        return

    if device == "GPU":
        bpy.context.scene.cycles.device = "GPU"
        return


def get_render_device_info():
    """获取渲染设备信息"""
    if not IN_BLENDER:
        return "N/A"

    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons.get("cycles")

    engine = bpy.context.scene.render.engine

    info = {
        "engine": engine,
        "device": "CPU",
        "gpu_devices": [],
    }

    if engine == "CYCLES" and cycles_prefs:
        cprefs = cycles_prefs.preferences
        info["device"] = bpy.context.scene.cycles.device

        try:
            compute_device_type = cprefs.compute_device_type
            info["compute_type"] = compute_device_type

            if hasattr(cprefs, "get_devices"):
                cprefs.get_devices()

            seen_devices = set()
            if hasattr(cprefs, "devices"):
                for device in cprefs.devices:
                    if device.use and device.type != "CPU":
                        if device.type == compute_device_type:
                            if device.name not in seen_devices:
                                info["gpu_devices"].append(device.name)
                                seen_devices.add(device.name)
        except Exception:
            pass

    elif engine in {"BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"}:
        info["device"] = "GPU"

    return info


def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}h {mins}m {secs:.0f}s"


def print_progress_bar(current, total, frame_time=None, elapsed=None,
                       prefix="渲染进度", use_cr=True):
    """打印进度条"""
    bar_length = 30
    progress = current / total if total > 0 else 0
    filled = int(bar_length * progress)
    bar = "█" * filled + "░" * (bar_length - filled)
    percent = progress * 100

    eta_str = ""
    if elapsed and current > 0:
        eta = (elapsed / current) * (total - current)
        eta_str = f" | ETA: {format_time(eta)}"

    frame_str = ""
    if frame_time:
        frame_str = f" | {format_time(frame_time)}/帧"

    progress_line = f"{prefix}: |{bar}| {percent:.1f}% ({current}/{total}){frame_str}{eta_str}"

    if use_cr and current < total:
        print(f"\r{progress_line}", end="", flush=True)
    else:
        print(f"\r{progress_line}")
        sys.stdout.flush()


def _select_camera(scene, camera_name):
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

    scene.camera = camera_obj
    return camera_obj


def _find_compositor_tree(scene):
    tree = None

    try:
        if hasattr(scene, "node_tree") and scene.node_tree:
            tree = scene.node_tree
    except Exception:
        pass

    if tree is None:
        try:
            compositor_trees = [
                ng for ng in bpy.data.node_groups if ng.bl_idname == "CompositorNodeTree"
            ]
            for ng in compositor_trees:
                if "composit" in ng.name.lower():
                    tree = ng
                    break
            if tree is None and compositor_trees:
                tree = compositor_trees[0]
        except Exception:
            pass

    return tree


def _trace_node_chain(node, visited=None):
    """递归遍历节点链，找到源头节点类型"""
    if visited is None:
        visited = set()
    if node in visited:
        return None
    visited.add(node)
    if node.type == "R_LAYERS":
        return "R_LAYERS"
    for input_slot in node.inputs:
        if input_slot.is_linked:
            for link in input_slot.links:
                result = _trace_node_chain(link.from_node, visited)
                if result:
                    return result
    return None


def _get_source_output_name(node, link_from_socket=None, visited=None):
    """获取节点链源头的输出名称"""
    if visited is None:
        visited = set()
    if node in visited:
        return None
    visited.add(node)
    if node.type == "R_LAYERS":
        if link_from_socket:
            return link_from_socket.name.lower()
        for output in node.outputs:
            if output.is_linked:
                return output.name.lower()
        return None
    for input_slot in node.inputs:
        if input_slot.is_linked:
            for link in input_slot.links:
                result = _get_source_output_name(link.from_node, link.from_socket, visited)
                if result:
                    return result
    return None


def _find_output_nodes(tree):
    rgb_file_output = None
    depth_file_output = None

    for node in tree.nodes:
        if node.type == "OUTPUT_FILE":
            for input_slot in node.inputs:
                if input_slot.is_linked:
                    link = input_slot.links[0]
                    source_output_name = _get_source_output_name(link.from_node, link.from_socket)
                    if not source_output_name:
                        source_output_name = link.from_socket.name.lower()
                    if "depth" in source_output_name or "z" in source_output_name or "v" in source_output_name:
                        depth_file_output = node
                        break
                    if "image" in source_output_name or "rgba" in source_output_name or "rgb" in source_output_name:
                        rgb_file_output = node
                        break
            if not any(slot.is_linked for slot in node.inputs):
                if rgb_file_output is None:
                    rgb_file_output = node
                elif depth_file_output is None:
                    depth_file_output = node

    return rgb_file_output, depth_file_output


def _set_file_output_path(node, directory, filename):
    if hasattr(node, "base_path"):
        node.base_path = directory + os.sep
    elif hasattr(node, "directory"):
        node.directory = directory + os.sep
    if hasattr(node, "file_slots") and len(node.file_slots) > 0:
        node.file_slots[0].path = filename
    elif hasattr(node, "file_output_items") and len(node.file_output_items) > 0:
        if hasattr(node, "file_name"):
            node.file_name = filename
        item = node.file_output_items[0]
        if hasattr(item, "name"):
            item.name = filename
    elif hasattr(node, "file_name"):
        node.file_name = filename


def _set_exr_format(format_obj):
    try:
        format_obj.file_format = "OPEN_EXR"
    except TypeError:
        format_obj.file_format = "OPEN_EXR_MULTILAYER"


def render_frames(blend_path: str, output_dir: str,
                  camera_name: str | None = None,
                  render_width: int | None = None,
                  render_height: int | None = None,
                  export_animation: bool = False,
                  frame_start: int | None = None,
                  frame_end: int | None = None,
                  frame_step: int = 1,
                  on_frame_rendered=None):
    """
    在 Blender 中渲染 RGB 和 Depth EXR
    """
    if not IN_BLENDER:
        raise RuntimeError("此函数必须在 Blender 环境中运行")

    blend_path = os.path.expanduser(blend_path)
    output_dir = os.path.abspath(os.path.expanduser(output_dir))

    print(f"\n{'=' * 60}")
    print("渲染任务")
    print(f"{'=' * 60}")
    print(f"  Blender 版本: {bpy.app.version_string}")
    sys.stdout.flush()

    print(f"  加载文件: {blend_path}")
    sys.stdout.flush()
    bpy.ops.wm.open_mainfile(filepath=blend_path)

    scene = bpy.context.scene
    view_layer = scene.view_layers[0]

    if render_width is None:
        render_width = scene.render.resolution_x
    if render_height is None:
        render_height = scene.render.resolution_y

    camera_obj = _select_camera(scene, camera_name)
    camera_data = camera_obj.data

    rgb_dir = os.path.join(output_dir, "rgb")
    depth_exr_dir = os.path.join(output_dir, "depth", "exr")

    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_exr_dir, exist_ok=True)

    view_layer.use_pass_z = True

    if export_animation:
        if frame_start is None:
            frame_start = scene.frame_start
        if frame_end is None:
            frame_end = scene.frame_end
    else:
        frame_start = scene.frame_current
        frame_end = scene.frame_current

    total_frames = len(range(frame_start, frame_end + 1, frame_step))

    apply_render_device(
        os.environ.get("FG_DEVICE"),
        os.environ.get("FG_COMPUTE_TYPE"),
    )

    print(f"  渲染引擎: {scene.render.engine}")
    device_info = get_render_device_info()
    if isinstance(device_info, dict):
        print(f"  渲染设备: {device_info['device']}")
        if device_info.get("gpu_devices"):
            for gpu in device_info["gpu_devices"]:
                print(f"    - {gpu}")
        if device_info.get("compute_type"):
            print(f"  计算类型: {device_info['compute_type']}")

    print(f"  分辨率: {render_width} x {render_height}")
    print(f"  帧范围: {frame_start} - {frame_end} (步长: {frame_step})")
    print(f"  总帧数: {total_frames}")
    print(f"  输出目录: {output_dir}")
    print(f"{'=' * 60}")
    print("")
    sys.stdout.flush()

    tree = _find_compositor_tree(scene)
    if tree:
        print(f"  合成器节点: {tree.name} ({len(tree.nodes)} nodes)")
    sys.stdout.flush()

    if not tree:
        raise RuntimeError("错误: 无法访问合成器节点树！")

    rgb_file_output, depth_file_output = _find_output_nodes(tree)
    if not rgb_file_output:
        raise RuntimeError("错误: 在合成器中没有找到 RGB 文件输出节点！")
    if not depth_file_output:
        raise RuntimeError("错误: 在合成器中没有找到 Depth 文件输出节点！")
    print("✓ 已找到 RGB 和 Depth 文件输出节点")
    sys.stdout.flush()

    frames_rendered = 0
    render_start_time = time.time()
    frame_times = []

    for frame in range(frame_start, frame_end + 1, frame_step):
        frame_start_time = time.time()
        frame_str = f"{frame:06d}"

        scene.frame_set(frame)
        bpy.context.view_layer.update()

        if rgb_file_output:
            _set_file_output_path(rgb_file_output, rgb_dir, frame_str)
            rgb_file_output.format.file_format = "PNG"

        if depth_file_output:
            _set_file_output_path(depth_file_output, depth_exr_dir, frame_str)
            _set_exr_format(depth_file_output.format)
            depth_file_output.format.color_depth = "32"
            try:
                depth_file_output.format.color_mode = "BW"
            except (AttributeError, TypeError):
                try:
                    depth_file_output.format.color_mode = "RGB"
                except Exception:
                    pass

        bpy.ops.render.render(write_still=False)

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

        if on_frame_rendered:
            on_frame_rendered(
                frame=frame,
                scene=scene,
                camera_obj=camera_obj,
                camera_data=camera_data,
                render_width=render_width,
                render_height=render_height,
                output_dir=output_dir,
            )

        frames_rendered += 1
        frame_elapsed = time.time() - frame_start_time
        frame_times.append(frame_elapsed)
        total_elapsed = time.time() - render_start_time
        avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
        print_progress_bar(frames_rendered, total_frames, avg_frame_time, total_elapsed)

    total_time = time.time() - render_start_time
    avg_time = total_time / frames_rendered if frames_rendered > 0 else 0

    print(f"\n\n{'=' * 60}")
    print("渲染完成!")
    print(f"{'=' * 60}")
    print(f"  总帧数: {frames_rendered}")
    print(f"  总用时: {format_time(total_time)}")
    print(f"  平均每帧: {format_time(avg_time)}")
    print("  输出目录:")
    print(f"    - RGB: {rgb_dir}")
    print(f"    - Depth EXR: {depth_exr_dir}")
    print(f"{'=' * 60}")

    return {
        "rgb_dir": rgb_dir,
        "depth_exr_dir": depth_exr_dir,
        "camera": camera_obj,
        "render_width": render_width,
        "render_height": render_height,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frame_step": frame_step,
    }
