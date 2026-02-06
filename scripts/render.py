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

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def apply_render_device(device: str | None, compute_type: str | None = None,
                        gpu_ids: list[int] | None = None) -> None:
    """
    根据配置设置 Blender 渲染设备
    device: CPU / GPU
    compute_type: CUDA / OPTIX / HIP / METAL / ONEAPI
    gpu_ids: 指定要使用的GPU索引列表，None表示使用所有GPU
    """
    if not IN_BLENDER or not device:
        return

    device = str(device).strip().upper()
    compute_type = str(compute_type).strip().upper() if compute_type else None

    # 检查是否启用详细输出
    verbose = os.environ.get("FG_VERBOSE", "0") == "1"

    if device == "CPU":
        bpy.context.scene.cycles.device = "CPU"
        if verbose:
            print("  渲染设备: CPU")
        return

    if device == "GPU":
        # 确保渲染引擎是 CYCLES（GPU 渲染需要 CYCLES）
        if bpy.context.scene.render.engine != "CYCLES":
            if verbose:
                print(f"  警告: 当前渲染引擎是 {bpy.context.scene.render.engine}，GPU 渲染需要 CYCLES")
                print(f"  正在切换到 CYCLES 引擎...")
            bpy.context.scene.render.engine = "CYCLES"
        
        bpy.context.scene.cycles.device = "GPU"

        # 将无效的 compute_type 视为未指定，走自动检测
        if compute_type in ("NONE", ""):
            compute_type = None

        try:
            cycles_prefs = bpy.context.preferences.addons.get("cycles")
            if not cycles_prefs:
                print("  错误: 未找到 Cycles 插件，无法使用 GPU，已回退到 CPU（会导致 CPU 占满）", file=sys.stderr)
                bpy.context.scene.cycles.device = "CPU"
                return

            cprefs = cycles_prefs.preferences

            # 如果没有指定 compute_type，尝试自动检测
            if not compute_type:
                # 按优先级尝试不同的计算类型（CUDA 优先，兼容性更好）
                for try_type in ["CUDA", "OPTIX", "HIP", "METAL", "ONEAPI"]:
                    try:
                        cprefs.compute_device_type = try_type
                        cprefs.get_devices()
                        # 检查是否有可用设备
                        has_device = any(
                            d.type == try_type for d in cprefs.devices
                        )
                        if has_device:
                            compute_type = try_type
                            if verbose:
                                print(f"  自动检测计算类型: {compute_type}")
                            break
                    except Exception:
                        continue

            if not compute_type:
                print("  错误: 未找到可用的 GPU 计算设备（请检查驱动/CUDA/Blender 是否支持 GPU），已回退到 CPU（会导致 CPU 占满）", file=sys.stderr)
                bpy.context.scene.cycles.device = "CPU"
                return

            # 设置计算类型并刷新设备列表
            cprefs.compute_device_type = compute_type
            if hasattr(cprefs, "get_devices"):
                cprefs.get_devices()

            # 启用 GPU 设备
            enabled_gpus = []
            gpu_index = 0

            if hasattr(cprefs, "devices"):
                for dev in cprefs.devices:
                    if dev.type == compute_type:
                        if gpu_ids is None:
                            # 使用所有 GPU
                            dev.use = True
                            enabled_gpus.append(f"{dev.name}")
                        else:
                            # 只使用指定的 GPU
                            dev.use = (gpu_index in gpu_ids)
                            if dev.use:
                                enabled_gpus.append(f"{dev.name}")
                        gpu_index += 1
                    elif dev.type == "CPU":
                        # 禁用 CPU 参与 GPU 渲染
                        dev.use = False

            if enabled_gpus:
                if verbose:
                    print(f"  计算类型: {compute_type}")
                    print(f"  启用 GPU ({len(enabled_gpus)}张):")
                    for gpu_name in enabled_gpus:
                        print(f"    - {gpu_name}")
            else:
                print(f"  错误: 未启用任何 GPU 设备（gpu_ids 或设备列表异常），已回退到 CPU（会导致 CPU 占满）", file=sys.stderr)
                bpy.context.scene.cycles.device = "CPU"

        except Exception as e:
            print(f"  错误: GPU 配置失败: {e}，已回退到 CPU（会导致 CPU 占满）", file=sys.stderr)
            bpy.context.scene.cycles.device = "CPU"


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


def _create_simple_compositor(scene, rgb_dir, depth_exr_dir):
    """创建简单的合成器节点树（不依赖用户预先配置）"""
    # 启用合成器
    scene.use_nodes = True
    scene.render.use_compositing = True
    tree = scene.node_tree
    
    # 清空现有节点
    for node in tree.nodes:
        tree.nodes.remove(node)
    
    # 创建 Render Layers 节点
    rl_node = tree.nodes.new(type="CompositorNodeRLayers")
    rl_node.location = (0, 0)
    
    # 创建 RGB 文件输出节点
    rgb_output = tree.nodes.new(type="CompositorNodeOutputFile")
    rgb_output.location = (400, 100)
    rgb_output.base_path = rgb_dir + os.sep
    rgb_output.format.file_format = "PNG"
    rgb_output.format.color_mode = "RGB"
    rgb_output.format.color_depth = "8"
    
    # 创建 Depth 文件输出节点
    depth_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_output.location = (400, -100)
    depth_output.base_path = depth_exr_dir + os.sep
    depth_output.format.file_format = "OPEN_EXR"
    depth_output.format.color_mode = "RGB"  # EXR 不支持 BW，使用 RGB
    depth_output.format.color_depth = "32"
    
    # 连接节点
    # RGB: Render Layers -> RGB Output
    tree.links.new(rl_node.outputs["Image"], rgb_output.inputs[0])
    
    # Depth: Render Layers -> Depth Output
    tree.links.new(rl_node.outputs["Depth"], depth_output.inputs[0])
    
    return rgb_output, depth_output


def render_frames_direct(blend_path: str, output_dir: str,
                         camera_name: str | None = None,
                         render_width: int | None = None,
                         render_height: int | None = None,
                         export_animation: bool = False,
                         frame_start: int | None = None,
                         frame_end: int | None = None,
                         frame_step: int = 1,
                         on_frame_rendered=None,
                         use_compositor: bool = False):
    """
    在 Blender 中渲染 RGB 和 Depth EXR（自动创建合成器节点，不依赖用户预先配置）
    
    Args:
        use_compositor: 如果为 True，则使用用户预先配置的合成器节点
    """
    if not IN_BLENDER:
        raise RuntimeError("此函数必须在 Blender 环境中运行")

    blend_path = os.path.expanduser(blend_path)
    output_dir = os.path.abspath(os.path.expanduser(output_dir))

    # 检查是否启用详细输出
    verbose = os.environ.get("FG_VERBOSE", "0") == "1"

    if verbose:
        print(f"\n{'=' * 60}")
        print("渲染任务（直接模式，自动创建合成器节点）")
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

    # 启用深度通道
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

    # 解析 GPU IDs（支持 "0,1,2" 或 "all" 格式）
    gpu_ids_str = os.environ.get("FG_GPU_IDS")
    gpu_ids = None
    if gpu_ids_str and gpu_ids_str.lower() != "all":
        try:
            gpu_ids = [int(x.strip()) for x in gpu_ids_str.split(",")]
        except ValueError:
            pass

    apply_render_device(
        os.environ.get("FG_DEVICE"),
        os.environ.get("FG_COMPUTE_TYPE"),
        gpu_ids,
    )

    # 若明确请求 GPU 但实际落到 CPU，直接失败，避免多进程把 CPU 拖死
    if os.environ.get("FG_DEVICE", "").upper() == "GPU" and getattr(scene.cycles, "device", "CPU") == "CPU":
        print("  错误: 已请求 GPU 渲染但未成功使用 GPU（见上方错误），当前为 CPU 渲染，已退出。", file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)

    # 始终输出关键信息（不受 verbose 控制）
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
    print(f"    - RGB: {rgb_dir}")
    print(f"    - Depth EXR: {depth_exr_dir}")
    print(f"{'=' * 60}")
    print("")
    sys.stdout.flush()

    # 如果不使用用户预先配置的合成器，则自动创建简单的合成器节点
    if not use_compositor:
        if verbose:
            print("  自动创建合成器节点...")
        rgb_file_output, depth_file_output = _create_simple_compositor(scene, rgb_dir, depth_exr_dir)
        if verbose:
            print("✓ 合成器节点已创建")
    else:
        # 使用用户预先配置的合成器节点
        tree = _find_compositor_tree(scene)
        if not tree:
            raise RuntimeError("错误: 无法访问合成器节点树！")
        rgb_file_output, depth_file_output = _find_output_nodes(tree)
        if not rgb_file_output:
            raise RuntimeError("错误: 在合成器中没有找到 RGB 文件输出节点！")
        if not depth_file_output:
            raise RuntimeError("错误: 在合成器中没有找到 Depth 文件输出节点！")
        if verbose:
            print("✓ 使用用户预先配置的合成器节点")

    frames_rendered = 0
    render_start_time = time.time()
    frame_times = []

    # 使用 tqdm 显示进度
    if HAS_TQDM:
        pbar = tqdm(
            total=total_frames,
            desc="渲染进度",
            unit="帧",
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
    else:
        pbar = None

    for frame in range(frame_start, frame_end + 1, frame_step):
        frame_start_time = time.time()
        frame_str = f"{frame:06d}"

        scene.frame_set(frame)
        bpy.context.view_layer.update()

        # 设置文件输出路径
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

        # 渲染
        bpy.ops.render.render(write_still=False)

        # 重命名文件（确保文件名格式正确）
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
        
        # 更新 tqdm 进度条
        if pbar is not None:
            pbar.update(1)
            # 更新描述信息（显示平均帧时间）
            if frame_times:
                avg_frame_time = sum(frame_times) / len(frame_times)
                pbar.set_postfix_str(f"{format_time(avg_frame_time)}/帧")
        else:
            # 回退到原来的进度条
            total_elapsed = time.time() - render_start_time
            avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
            print_progress_bar(frames_rendered, total_frames, avg_frame_time, total_elapsed)

    # 关闭进度条
    if pbar is not None:
        pbar.close()

    total_time = time.time() - render_start_time
    avg_time = total_time / frames_rendered if frames_rendered > 0 else 0

    if verbose:
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


def render_frames(blend_path: str, output_dir: str,
                  camera_name: str | None = None,
                  render_width: int | None = None,
                  render_height: int | None = None,
                  export_animation: bool = False,
                  frame_start: int | None = None,
                  frame_end: int | None = None,
                  frame_step: int = 1,
                  on_frame_rendered=None,
                  use_compositor: bool = True):
    """
    在 Blender 中渲染 RGB 和 Depth EXR（使用合成器节点）
    
    Args:
        use_compositor: 如果为 False，则使用直接模式（不依赖合成器节点）
    """
    # 如果不需要合成器，使用直接模式
    if not use_compositor:
        return render_frames_direct(
            blend_path, output_dir, camera_name, render_width, render_height,
            export_animation, frame_start, frame_end, frame_step, on_frame_rendered, use_compositor=False
        )
    if not IN_BLENDER:
        raise RuntimeError("此函数必须在 Blender 环境中运行")

    blend_path = os.path.expanduser(blend_path)
    output_dir = os.path.abspath(os.path.expanduser(output_dir))

    # 检查是否启用详细输出
    verbose = os.environ.get("FG_VERBOSE", "0") == "1"

    if verbose:
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

    # 解析 GPU IDs（支持 "0,1,2" 或 "all" 格式）
    gpu_ids_str = os.environ.get("FG_GPU_IDS")
    gpu_ids = None
    if gpu_ids_str and gpu_ids_str.lower() != "all":
        try:
            gpu_ids = [int(x.strip()) for x in gpu_ids_str.split(",")]
        except ValueError:
            pass

    apply_render_device(
        os.environ.get("FG_DEVICE"),
        os.environ.get("FG_COMPUTE_TYPE"),
        gpu_ids,
    )

    # 若明确请求 GPU 但实际落到 CPU，直接失败，避免多进程把 CPU 拖死
    if os.environ.get("FG_DEVICE", "").upper() == "GPU" and getattr(scene.cycles, "device", "CPU") == "CPU":
        print("  错误: 已请求 GPU 渲染但未成功使用 GPU（见上方错误），当前为 CPU 渲染，已退出。", file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)

    # 始终输出关键信息（不受 verbose 控制）
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
    print(f"    - RGB: {rgb_dir}")
    print(f"    - Depth EXR: {depth_exr_dir}")
    print(f"{'=' * 60}")
    print("")
    sys.stdout.flush()

    tree = _find_compositor_tree(scene)
    if tree and verbose:
        print(f"  合成器节点: {tree.name} ({len(tree.nodes)} nodes)")
        sys.stdout.flush()

    if not tree:
        raise RuntimeError("错误: 无法访问合成器节点树！")

    rgb_file_output, depth_file_output = _find_output_nodes(tree)
    if not rgb_file_output:
        raise RuntimeError("错误: 在合成器中没有找到 RGB 文件输出节点！")
    if not depth_file_output:
        raise RuntimeError("错误: 在合成器中没有找到 Depth 文件输出节点！")
    if verbose:
        print("✓ 已找到 RGB 和 Depth 文件输出节点")
        sys.stdout.flush()

    frames_rendered = 0
    render_start_time = time.time()
    frame_times = []

    # 使用 tqdm 显示进度
    if HAS_TQDM:
        pbar = tqdm(
            total=total_frames,
            desc="渲染进度",
            unit="帧",
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
    else:
        pbar = None

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
        
        # 更新 tqdm 进度条
        if pbar is not None:
            pbar.update(1)
            # 更新描述信息（显示平均帧时间）
            if frame_times:
                avg_frame_time = sum(frame_times) / len(frame_times)
                pbar.set_postfix_str(f"{format_time(avg_frame_time)}/帧")
        else:
            # 回退到原来的进度条
            total_elapsed = time.time() - render_start_time
            avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
            print_progress_bar(frames_rendered, total_frames, avg_frame_time, total_elapsed)

    # 关闭进度条
    if pbar is not None:
        pbar.close()

    total_time = time.time() - render_start_time
    avg_time = total_time / frames_rendered if frames_rendered > 0 else 0

    if verbose:
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
