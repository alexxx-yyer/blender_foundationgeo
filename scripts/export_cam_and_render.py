#!/usr/bin/env python3
"""从 Blender .blend 文件中提取相机参数并导出 focal.txt 和 pose.txt"""

import os
import sys
import argparse
import numpy as np

# 尝试导入 bpy，如果失败则提示用户
try:
    import bpy
except ImportError:
    print("错误: 无法导入 bpy 模块")
    print("请确保在 Blender 环境中运行此脚本，或使用以下命令:")
    print("  blender --background --python export_cam_and_render.py -- <blend_file> [options]")
    sys.exit(1)


def get_camera_intrinsics(camera, render_width, render_height):
    """
    计算相机内参（焦距）
    
    Args:
        camera: Blender 相机对象
        render_width: 渲染宽度
        render_height: 渲染高度
    
    Returns:
        focal_length: 焦距（像素单位）
    """
    # 获取传感器尺寸（毫米）
    sensor_width = camera.sensor_width
    sensor_height = camera.sensor_height
    
    # 获取焦距（毫米）
    focal_length_mm = camera.lens
    
    # 计算焦距（像素单位）
    # 使用传感器宽度和渲染宽度
    focal_length_px = (focal_length_mm / sensor_width) * render_width
    
    # 如果传感器高度和渲染高度不同，也计算 fy
    focal_length_py = (focal_length_mm / sensor_height) * render_height
    
    return focal_length_px, focal_length_py


def get_camera_pose(camera, use_evaluated=True):
    """
    获取相机位姿（4x4 变换矩阵，从世界坐标到相机坐标）
    
    Args:
        camera: Blender 相机对象
        use_evaluated: 是否使用评估后的对象（用于动画）
    
    Returns:
        pose_matrix: 4x4 numpy 数组，表示从世界坐标到相机坐标的变换
    """
    # 获取相机的世界变换矩阵
    # Blender 的矩阵是从世界坐标到对象坐标的变换
    # 我们需要的是从世界坐标到相机坐标的变换
    
    if use_evaluated:
        try:
            # 尝试使用依赖图获取评估后的对象（包含动画数据）
            # 在交互模式下使用 evaluated_depsgraph_get()
            if hasattr(bpy.context, 'evaluated_depsgraph_get'):
                depsgraph = bpy.context.evaluated_depsgraph_get()
                camera_eval = camera.evaluated_get(depsgraph)
                matrix_world = camera_eval.matrix_world
            # 在背景模式下使用 depsgraph
            elif hasattr(bpy.context, 'depsgraph'):
                depsgraph = bpy.context.depsgraph
                camera_eval = camera.evaluated_get(depsgraph)
                matrix_world = camera_eval.matrix_world
            else:
                # 回退到直接获取（可能不包含动画）
                matrix_world = camera.matrix_world
        except:
            # 如果评估失败，回退到直接获取
            matrix_world = camera.matrix_world
    else:
        matrix_world = camera.matrix_world
    
    # 转换为 numpy 数组
    pose = np.array(matrix_world)
    
    # 相机坐标系：Blender 中相机朝向 -Z 方向，上方向为 +Y
    # 标准计算机视觉坐标系：相机朝向 +Z 方向，上方向为 -Y
    # 需要进行坐标系转换
    
    # 创建坐标系转换矩阵（从 Blender 坐标系到 CV 坐标系）
    # Blender: X右 Y前 Z上 -> CV: X右 Y下 Z前
    coord_transform = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    
    # 应用坐标系转换
    pose_cv = coord_transform @ pose
    
    return pose_cv


def export_camera_data(blend_path: str, output_dir: str | None = None, 
                       camera_name: str | None = None, 
                       render_width: int | None = None,
                       render_height: int | None = None,
                       only_camera: bool = False,
                       list_cameras: bool = False,
                       export_animation: bool = False,
                       frame_start: int | None = None,
                       frame_end: int | None = None,
                       frame_step: int = 1):
    """
    从 Blender 文件中导出相机数据
    
    Args:
        blend_path: .blend 文件路径
        output_dir: 输出目录，如果为 None 则使用 .blend 文件所在目录
        camera_name: 相机名称，如果为 None 则使用活动相机
        render_width: 渲染宽度，如果为 None 则使用场景设置
        render_height: 渲染高度，如果为 None 则使用场景设置
        only_camera: 如果为 True，只列出相机信息，不导出文件
        list_cameras: 如果为 True，列出所有可用的相机名称并退出
        export_animation: 如果为 True，导出动画中每一帧的相机参数
        frame_start: 起始帧（默认：使用场景设置）
        frame_end: 结束帧（默认：使用场景设置）
        frame_step: 帧步长（默认：1，即每一帧）
    """
    blend_path = os.path.expanduser(blend_path)
    
    if not os.path.exists(blend_path):
        raise FileNotFoundError(f"找不到文件: {blend_path}")
    
    # 打开 Blender 文件
    bpy.ops.wm.open_mainfile(filepath=blend_path)
    
    # 获取场景
    scene = bpy.context.scene
    
    # 列出所有可用的相机
    all_cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    
    # 如果只是列出相机，则打印并退出
    if list_cameras:
        print(f"场景中找到 {len(all_cameras)} 个相机:")
        if all_cameras:
            for i, cam in enumerate(all_cameras, 1):
                is_active = " (活动相机)" if cam == scene.camera else ""
                print(f"  {i}. {cam.name}{is_active}")
        else:
            print("  没有找到相机对象")
        return None
    
    # 获取渲染尺寸
    if render_width is None:
        render_width = scene.render.resolution_x
    if render_height is None:
        render_height = scene.render.resolution_y
    
    # 获取相机
    if camera_name is None:
        camera = scene.camera
        if camera is None:
            if all_cameras:
                print(f"场景中没有活动相机，但找到 {len(all_cameras)} 个相机对象:")
                for cam in all_cameras:
                    print(f"  - {cam.name}")
                raise ValueError("请使用 -c 参数指定相机名称")
            else:
                raise ValueError("场景中没有相机对象")
    else:
        # 尝试精确匹配
        camera = bpy.data.objects.get(camera_name)
        
        # 如果精确匹配失败，尝试模糊匹配（不区分大小写）
        if camera is None or camera.type != 'CAMERA':
            camera = None
            camera_name_lower = camera_name.lower()
            for obj in all_cameras:
                if obj.name.lower() == camera_name_lower or camera_name_lower in obj.name.lower():
                    camera = obj
                    print(f"找到匹配的相机: {obj.name} (搜索: {camera_name})")
                    break
        
        if camera is None or camera.type != 'CAMERA':
            if all_cameras:
                print(f"找不到名为 '{camera_name}' 的相机对象")
                print(f"可用的相机:")
                for cam in all_cameras:
                    print(f"  - {cam.name}")
            raise ValueError(f"找不到名为 '{camera_name}' 的相机对象")
    
    # 获取相机数据
    camera_data = camera.data
    
    # 如果导出动画，遍历每一帧
    if export_animation:
        # 获取帧范围
        if frame_start is None:
            frame_start = scene.frame_start
        if frame_end is None:
            frame_end = scene.frame_end
        
        print(f"检测到动画，导出帧范围: {frame_start} - {frame_end} (步长: {frame_step})")
        
        # 确定输出目录
        if output_dir is None:
            output_dir = os.path.dirname(blend_path)
        else:
            output_dir = os.path.expanduser(output_dir)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 遍历每一帧
        frames_exported = 0
        for frame in range(frame_start, frame_end + 1, frame_step):
            # 设置当前帧
            scene.frame_set(frame)
            
            # 更新依赖图以确保变换矩阵正确
            # 在背景模式下，需要手动更新依赖图
            bpy.context.view_layer.update()
            
            # 强制更新场景的动画数据
            # 这确保动画关键帧被正确评估
            scene.update_tag()
            
            # 在背景模式下，需要手动更新依赖图
            # 获取或创建依赖图并更新
            try:
                # 尝试获取依赖图
                if hasattr(bpy.context, 'depsgraph') and bpy.context.depsgraph:
                    depsgraph = bpy.context.depsgraph
                    depsgraph.update()
                elif hasattr(bpy.context, 'evaluated_depsgraph_get'):
                    depsgraph = bpy.context.evaluated_depsgraph_get()
                    depsgraph.update()
                elif hasattr(bpy.context.view_layer, 'depsgraph'):
                    depsgraph = bpy.context.view_layer.depsgraph
                    depsgraph.update()
            except Exception as e:
                # 如果更新失败，继续使用原始对象
                pass
            
            # 计算焦距（焦距通常不变，但为了完整性每次都计算）
            fx, fy = get_camera_intrinsics(camera_data, render_width, render_height)
            
            # 获取当前帧的相机位姿（使用评估后的对象以获取动画数据）
            pose = get_camera_pose(camera, use_evaluated=True)
            
            # 为每一帧创建单独的文件
            frame_str = f"{frame:06d}"
            focal_path = os.path.join(output_dir, f'focal_{frame_str}.txt')
            pose_path = os.path.join(output_dir, f'pose_{frame_str}.txt')
            
            # 保存 focal.txt
            with open(focal_path, 'w') as f:
                if abs(fx - fy) < 1e-6:
                    f.write(f"{fx:.6f}\n")
                else:
                    f.write(f"{fx:.6f} {fy:.6f}\n")
            
            # 保存 pose.txt
            np.savetxt(pose_path, pose, fmt='%.8f')
            
            # 打印调试信息（前几帧和最后几帧）
            if frames_exported < 3 or frame >= frame_end - 2:
                print(f"  帧 {frame}: 位置=({camera.location.x:.3f}, {camera.location.y:.3f}, {camera.location.z:.3f})")
            
            frames_exported += 1
            if frames_exported % 10 == 0 or frame == frame_end:
                print(f"  已导出帧 {frame}/{frame_end} ({frames_exported} 帧)")
        
        print(f"\n动画导出完成!")
        print(f"  总帧数: {frames_exported}")
        print(f"  输出目录: {output_dir}")
        print(f"  文件格式: focal_XXXXXX.txt, pose_XXXXXX.txt")
        
        return None
    
    # 单帧导出（原有逻辑）
    # 计算焦距
    fx, fy = get_camera_intrinsics(camera_data, render_width, render_height)
    
    # 获取相机位姿
    pose = get_camera_pose(camera)
    
    # 打印相机信息
    print(f"相机信息:")
    print(f"  输入文件: {blend_path}")
    print(f"  相机名称: {camera.name}")
    print(f"  渲染尺寸: {render_width} x {render_height}")
    print(f"  焦距 (fx, fy): ({fx:.6f}, {fy:.6f})")
    print(f"  焦距 (mm): {camera_data.lens:.2f}")
    print(f"  传感器尺寸: {camera_data.sensor_width:.2f} x {camera_data.sensor_height:.2f} mm")
    print(f"  相机位置: ({camera.location.x:.4f}, {camera.location.y:.4f}, {camera.location.z:.4f})")
    print(f"  相机旋转: ({camera.rotation_euler.x:.4f}, {camera.rotation_euler.y:.4f}, {camera.rotation_euler.z:.4f})")
    
    # 如果 only_camera 为 True，只显示信息，不导出文件
    if only_camera:
        print(f"\n位姿矩阵 (4x4):")
        for i in range(4):
            print(f"  [{pose[i,0]:.6f} {pose[i,1]:.6f} {pose[i,2]:.6f} {pose[i,3]:.6f}]")
        return fx, fy, pose
    
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.dirname(blend_path)
    else:
        output_dir = os.path.expanduser(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存 focal.txt
    focal_path = os.path.join(output_dir, 'focal.txt')
    with open(focal_path, 'w') as f:
        # 通常保存 fx 和 fy，如果两者相同也可以只保存一个值
        if abs(fx - fy) < 1e-6:
            f.write(f"{fx:.6f}\n")
        else:
            f.write(f"{fx:.6f} {fy:.6f}\n")
    
    # 保存 pose.txt (4x4 矩阵)
    pose_path = os.path.join(output_dir, 'pose.txt')
    np.savetxt(pose_path, pose, fmt='%.8f')
    
    print(f"\n导出完成!")
    print(f"  输出文件:")
    print(f"    - {focal_path}")
    print(f"    - {pose_path}")
    
    return fx, fy, pose


if __name__ == "__main__":
    # 在 Blender 中运行时，需要处理命令行参数
    # Blender 会传递 '--' 作为分隔符
    
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        argv = sys.argv[1:]
    
    parser = argparse.ArgumentParser(
        description="从 Blender .blend 文件中导出相机参数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 在 Blender 中运行:
  blender --background --python export_cam_and_render.py -- input.blend
  
  # 指定输出目录和相机:
  blender --background --python export_cam_and_render.py -- input.blend -o output/ -c Camera
  
  # 指定渲染尺寸:
  blender --background --python export_cam_and_render.py -- input.blend -w 1920 --height 1080
  
  # 列出所有可用的相机:
  blender --background --python export_cam_and_render.py -- input.blend --list-cameras
  
  # 只显示相机信息，不导出文件:
  blender --background --python export_cam_and_render.py -- input.blend --only-camera
  
  # 导出动画中每一帧的相机参数:
  blender --background --python export_cam_and_render.py -- input.blend --export-animation
  
  # 导出指定帧范围的动画:
  blender --background --python export_cam_and_render.py -- input.blend --export-animation --frame-start 1 --frame-end 100 --frame-step 5
        """
    )
    
    parser.add_argument("blend_file", help="输入的 .blend 文件路径")
    parser.add_argument("-o", "--output", help="输出目录（默认：与 .blend 文件同目录）")
    parser.add_argument("-c", "--camera", help="相机名称（默认：活动相机）")
    parser.add_argument("-w", "--width", type=int, help="渲染宽度（默认：使用场景设置）")
    parser.add_argument("--height", type=int, help="渲染高度（默认：使用场景设置）")
    parser.add_argument("--only-camera", action="store_true", 
                       help="只显示相机信息，不导出文件")
    parser.add_argument("--list-cameras", action="store_true",
                       help="列出场景中所有可用的相机名称")
    parser.add_argument("--export-animation", action="store_true",
                       help="导出动画中每一帧的相机参数")
    parser.add_argument("--frame-start", type=int, default=None,
                       help="起始帧（默认：使用场景设置）")
    parser.add_argument("--frame-end", type=int, default=None,
                       help="结束帧（默认：使用场景设置）")
    parser.add_argument("--frame-step", type=int, default=1,
                       help="帧步长（默认：1，即每一帧）")
    
    args = parser.parse_args(argv)
    
    try:
        export_camera_data(
            args.blend_file,
            args.output,
            args.camera,
            args.width,
            args.height,
            args.only_camera,
            args.list_cameras,
            args.export_animation,
            args.frame_start,
            args.frame_end,
            args.frame_step
        )
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
