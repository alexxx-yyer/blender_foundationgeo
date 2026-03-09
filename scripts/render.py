#!/usr/bin/env python3
"""Blender ： RGB  EXR"""

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
     Blender Render device
    device: CPU / GPU
    compute_type: CUDA / OPTIX / HIP / METAL / ONEAPI
    gpu_ids: GPU，NoneGPU
    """
    if not IN_BLENDER or not device:
        return

    device = str(device).strip().upper()
    compute_type = str(compute_type).strip().upper() if compute_type else None

    # enable
    verbose = os.environ.get("FG_VERBOSE", "0") == "1"

    if device == "CPU":
        bpy.context.scene.cycles.device = "CPU"
        if verbose:
            print("  Render device: CPU")
        return

    if device == "GPU":
        # Render engine CYCLES（GPU  CYCLES）
        if bpy.context.scene.render.engine != "CYCLES":
            if verbose:
                print(f"  Warning: Render engine {bpy.context.scene.render.engine}，GPU  CYCLES")
                print(f"   CYCLES ...")
            bpy.context.scene.render.engine = "CYCLES"
        
        bpy.context.scene.cycles.device = "GPU"

        #  compute_type ，auto-detect
        if compute_type in ("NONE", ""):
            compute_type = None

        try:
            cycles_prefs = bpy.context.preferences.addons.get("cycles")
            if not cycles_prefs:
                print("  Error:  Cycles ， GPU， CPU（ CPU ）", file=sys.stderr)
                bpy.context.scene.cycles.device = "CPU"
                return

            cprefs = cycles_prefs.preferences

            #  compute_type，auto-detect
            if not compute_type:
                # Compute type（CUDA ，）
                for try_type in ["CUDA", "OPTIX", "HIP", "METAL", "ONEAPI"]:
                    try:
                        cprefs.compute_device_type = try_type
                        cprefs.get_devices()
                        # 
                        has_device = any(
                            d.type == try_type for d in cprefs.devices
                        )
                        if has_device:
                            compute_type = try_type
                            if verbose:
                                print(f"  auto-detectCompute type: {compute_type}")
                            break
                    except Exception:
                        continue

            if not compute_type:
                print("  Error:  GPU （/CUDA/Blender  GPU）， CPU（ CPU ）", file=sys.stderr)
                bpy.context.scene.cycles.device = "CPU"
                return

            # Compute type
            cprefs.compute_device_type = compute_type
            if hasattr(cprefs, "get_devices"):
                cprefs.get_devices()

            # enable GPU 
            enabled_gpus = []
            gpu_index = 0

            if hasattr(cprefs, "devices"):
                for dev in cprefs.devices:
                    if dev.type == compute_type:
                        if gpu_ids is None:
                            #  GPU
                            dev.use = True
                            enabled_gpus.append(f"{dev.name}")
                        else:
                            #  GPU
                            dev.use = (gpu_index in gpu_ids)
                            if dev.use:
                                enabled_gpus.append(f"{dev.name}")
                        gpu_index += 1
                    elif dev.type == "CPU":
                        # disable CPU  GPU 
                        dev.use = False

            if enabled_gpus:
                if verbose:
                    print(f"  Compute type: {compute_type}")
                    print(f"  enable GPU ({len(enabled_gpus)}):")
                    for gpu_name in enabled_gpus:
                        print(f"    - {gpu_name}")
            else:
                print(f"  Error: enable GPU （gpu_ids ）， CPU（ CPU ）", file=sys.stderr)
                bpy.context.scene.cycles.device = "CPU"

        except Exception as e:
            print(f"  Error: GPU Failed: {e}， CPU（ CPU ）", file=sys.stderr)
            bpy.context.scene.cycles.device = "CPU"


def get_render_device_info():
    """Render device"""
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
    """"""
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
                       prefix="Render progress", use_cr=True):
    """"""
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
        frame_str = f" | {format_time(frame_time)}/"

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
                raise ValueError("SceneCamera， -c Camera")
            raise ValueError("SceneCamera")
    else:
        camera_obj = bpy.data.objects.get(camera_name)
        if camera_obj is None or camera_obj.type != "CAMERA":
            camera_name_lower = camera_name.lower()
            for obj in all_cameras:
                if obj.name.lower() == camera_name_lower or camera_name_lower in obj.name.lower():
                    camera_obj = obj
                    print(f"Camera: {obj.name} (: {camera_name})")
                    break

        if camera_obj is None or camera_obj.type != "CAMERA":
            raise ValueError(f" '{camera_name}' Camera")

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
    """，"""
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
    """"""
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
    """（）"""
    # enable
    scene.use_nodes = True
    scene.render.use_compositing = True
    tree = scene.node_tree
    
    # 
    for node in tree.nodes:
        tree.nodes.remove(node)
    
    #  Render Layers 
    rl_node = tree.nodes.new(type="CompositorNodeRLayers")
    rl_node.location = (0, 0)
    
    #  RGB file
    rgb_output = tree.nodes.new(type="CompositorNodeOutputFile")
    rgb_output.location = (400, 100)
    rgb_output.base_path = rgb_dir + os.sep
    rgb_output.format.file_format = "PNG"
    rgb_output.format.color_mode = "RGB"
    rgb_output.format.color_depth = "8"
    
    #  Depth file
    depth_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_output.location = (400, -100)
    depth_output.base_path = depth_exr_dir + os.sep
    depth_output.format.file_format = "OPEN_EXR"
    depth_output.format.color_mode = "RGB"  # EXR  BW， RGB
    depth_output.format.color_depth = "32"
    
    # 
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
     Blender  RGB  Depth EXR（，）
    
    Args:
        use_compositor:  True，
    """
    if not IN_BLENDER:
        raise RuntimeError(" Blender ")

    blend_path = os.path.expanduser(blend_path)
    output_dir = os.path.abspath(os.path.expanduser(output_dir))

    # enable
    verbose = os.environ.get("FG_VERBOSE", "0") == "1"

    if verbose:
        print(f"\n{'=' * 60}")
        print("（，）")
        print(f"{'=' * 60}")
        print(f"  Blender : {bpy.app.version_string}")
        sys.stdout.flush()

        print(f"  file: {blend_path}")
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

    # enable
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

    #  GPU IDs（ "0,1,2"  "all" ）
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

    #  GPU  CPU，Failed， CPU 
    if os.environ.get("FG_DEVICE", "").upper() == "GPU" and getattr(scene.cycles, "device", "CPU") == "CPU":
        print("  Error:  GPU  GPU（Error）， CPU ，。", file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)

    # （ verbose ）
    print(f"  Render engine: {scene.render.engine}")
    device_info = get_render_device_info()
    if isinstance(device_info, dict):
        print(f"  Render device: {device_info['device']}")
        if device_info.get("gpu_devices"):
            for gpu in device_info["gpu_devices"]:
                print(f"    - {gpu}")
        if device_info.get("compute_type"):
            print(f"  Compute type: {device_info['compute_type']}")
    print(f"  Resolution: {render_width} x {render_height}")
    print(f"  : {frame_start} - {frame_end} (: {frame_step})")
    print(f"  Total frames: {total_frames}")
    print(f"  Output directory: {output_dir}")
    print(f"    - RGB: {rgb_dir}")
    print(f"    - Depth EXR: {depth_exr_dir}")
    print(f"{'=' * 60}")
    print("")
    sys.stdout.flush()

    # ，
    if not use_compositor:
        if verbose:
            print("  ...")
        rgb_file_output, depth_file_output = _create_simple_compositor(scene, rgb_dir, depth_exr_dir)
        if verbose:
            print("✓ ")
    else:
        # 
        tree = _find_compositor_tree(scene)
        if not tree:
            raise RuntimeError("Error: ！")
        rgb_file_output, depth_file_output = _find_output_nodes(tree)
        if not rgb_file_output:
            raise RuntimeError("Error:  RGB file！")
        if not depth_file_output:
            raise RuntimeError("Error:  Depth file！")
        if verbose:
            print("✓ ")

    frames_rendered = 0
    render_start_time = time.time()
    frame_times = []

    #  tqdm 
    if HAS_TQDM:
        pbar = tqdm(
            total=total_frames,
            desc="Render progress",
            unit="",
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

        # filepath
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

        # 
        bpy.ops.render.render(write_still=False)

        # file（file）
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
        
        #  tqdm 
        if pbar is not None:
            pbar.update(1)
            # （）
            if frame_times:
                avg_frame_time = sum(frame_times) / len(frame_times)
                pbar.set_postfix_str(f"{format_time(avg_frame_time)}/")
        else:
            # 
            total_elapsed = time.time() - render_start_time
            avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
            print_progress_bar(frames_rendered, total_frames, avg_frame_time, total_elapsed)

    # 
    if pbar is not None:
        pbar.close()

    total_time = time.time() - render_start_time
    avg_time = total_time / frames_rendered if frames_rendered > 0 else 0

    if verbose:
        print(f"\n\n{'=' * 60}")
        print("Done!")
        print(f"{'=' * 60}")
        print(f"  Total frames: {frames_rendered}")
        print(f"  Total time: {format_time(total_time)}")
        print(f"  Avg/frame: {format_time(avg_time)}")
        print("  Output directory:")
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
     Blender  RGB  Depth EXR（）
    
    Args:
        use_compositor:  False，（）
    """
    # ，
    if not use_compositor:
        return render_frames_direct(
            blend_path, output_dir, camera_name, render_width, render_height,
            export_animation, frame_start, frame_end, frame_step, on_frame_rendered, use_compositor=False
        )
    if not IN_BLENDER:
        raise RuntimeError(" Blender ")

    blend_path = os.path.expanduser(blend_path)
    output_dir = os.path.abspath(os.path.expanduser(output_dir))

    # enable
    verbose = os.environ.get("FG_VERBOSE", "0") == "1"

    if verbose:
        print(f"\n{'=' * 60}")
        print("")
        print(f"{'=' * 60}")
        print(f"  Blender : {bpy.app.version_string}")
        sys.stdout.flush()

        print(f"  file: {blend_path}")
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

    #  GPU IDs（ "0,1,2"  "all" ）
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

    #  GPU  CPU，Failed， CPU 
    if os.environ.get("FG_DEVICE", "").upper() == "GPU" and getattr(scene.cycles, "device", "CPU") == "CPU":
        print("  Error:  GPU  GPU（Error）， CPU ，。", file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)

    # （ verbose ）
    print(f"  Render engine: {scene.render.engine}")
    device_info = get_render_device_info()
    if isinstance(device_info, dict):
        print(f"  Render device: {device_info['device']}")
        if device_info.get("gpu_devices"):
            for gpu in device_info["gpu_devices"]:
                print(f"    - {gpu}")
        if device_info.get("compute_type"):
            print(f"  Compute type: {device_info['compute_type']}")

    print(f"  Resolution: {render_width} x {render_height}")
    print(f"  : {frame_start} - {frame_end} (: {frame_step})")
    print(f"  Total frames: {total_frames}")
    print(f"  Output directory: {output_dir}")
    print(f"    - RGB: {rgb_dir}")
    print(f"    - Depth EXR: {depth_exr_dir}")
    print(f"{'=' * 60}")
    print("")
    sys.stdout.flush()

    tree = _find_compositor_tree(scene)
    if tree and verbose:
        print(f"  : {tree.name} ({len(tree.nodes)} nodes)")
        sys.stdout.flush()

    if not tree:
        raise RuntimeError("Error: ！")

    rgb_file_output, depth_file_output = _find_output_nodes(tree)
    if not rgb_file_output:
        raise RuntimeError("Error:  RGB file！")
    if not depth_file_output:
        raise RuntimeError("Error:  Depth file！")
    if verbose:
        print("✓  RGB  Depth file")
        sys.stdout.flush()

    frames_rendered = 0
    render_start_time = time.time()
    frame_times = []

    #  tqdm 
    if HAS_TQDM:
        pbar = tqdm(
            total=total_frames,
            desc="Render progress",
            unit="",
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
        
        #  tqdm 
        if pbar is not None:
            pbar.update(1)
            # （）
            if frame_times:
                avg_frame_time = sum(frame_times) / len(frame_times)
                pbar.set_postfix_str(f"{format_time(avg_frame_time)}/")
        else:
            # 
            total_elapsed = time.time() - render_start_time
            avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
            print_progress_bar(frames_rendered, total_frames, avg_frame_time, total_elapsed)

    # 
    if pbar is not None:
        pbar.close()

    total_time = time.time() - render_start_time
    avg_time = total_time / frames_rendered if frames_rendered > 0 else 0

    if verbose:
        print(f"\n\n{'=' * 60}")
        print("Done!")
        print(f"{'=' * 60}")
        print(f"  Total frames: {frames_rendered}")
        print(f"  Total time: {format_time(total_time)}")
        print(f"  Avg/frame: {format_time(avg_time)}")
        print("  Output directory:")
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
