#!/usr/bin/env python3
"""批量从 Blender .blend 文件中导出相机参数"""

import os
import sys
import glob
import subprocess
import argparse


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


def batch_export_cameras(blend_dir: str, output_base_dir: str | None = None,
                         camera_name: str | None = None,
                         render_width: int | None = None,
                         render_height: int | None = None,
                         only_camera: bool = False,
                         export_animation: bool = False,
                         frame_start: int | None = None,
                         frame_end: int | None = None,
                         frame_step: int = 1):
    """
    批量处理目录中的所有 .blend 文件
    
    Args:
        blend_dir: 包含 .blend 文件的目录
        output_base_dir: 输出基础目录，如果为 None 则在每个 .blend 文件同目录输出
        camera_name: 相机名称
        render_width: 渲染宽度
        render_height: 渲染高度
    """
    blend_dir = os.path.expanduser(blend_dir)
    
    if not os.path.isdir(blend_dir):
        raise NotADirectoryError(f"目录不存在: {blend_dir}")
    
    # 查找所有 .blend 文件（不包括 .blend1 等备份文件）
    blend_files = glob.glob(os.path.join(blend_dir, '*.blend'))
    blend_files = [f for f in blend_files if not f.endswith('.blend1')]
    
    if not blend_files:
        print(f"在目录 {blend_dir} 中未找到 .blend 文件")
        return
    
    print(f"找到 {len(blend_files)} 个 .blend 文件")
    
    # 查找 Blender 可执行文件
    blender_exe = find_blender_executable()
    if blender_exe is None:
        print("错误: 找不到 Blender 可执行文件")
        print("请确保 Blender 已安装并在 PATH 中，或使用 --blender 参数指定路径")
        return
    
    print(f"使用 Blender: {blender_exe}")
    
    # 获取脚本路径
    script_path = os.path.join(
        os.path.dirname(__file__),
        'export_cam_and_render.py'
    )
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"找不到脚本: {script_path}")
    
    # 处理每个文件
    success_count = 0
    fail_count = 0
    
    for blend_file in sorted(blend_files):
        print(f"\n处理: {os.path.basename(blend_file)}")
        
        # 确定输出目录
        if output_base_dir:
            # 使用输出基础目录 + 文件名（不含扩展名）
            base_name = os.path.splitext(os.path.basename(blend_file))[0]
            output_dir = os.path.join(output_base_dir, base_name)
        else:
            # 使用 .blend 文件所在目录
            output_dir = os.path.dirname(blend_file)
        
        # 构建命令
        cmd = [
            blender_exe,
            '--background',
            '--python', script_path,
            '--',
            blend_file
        ]
        
        if output_dir:
            cmd.extend(['-o', output_dir])
        if camera_name:
            cmd.extend(['-c', camera_name])
        if render_width:
            cmd.extend(['-w', str(render_width)])
        if render_height:
            cmd.extend(['--height', str(render_height)])
        if only_camera:
            cmd.append('--only-camera')
        if export_animation:
            cmd.append('--export-animation')
            if frame_start is not None:
                cmd.extend(['--frame-start', str(frame_start)])
            if frame_end is not None:
                cmd.extend(['--frame-end', str(frame_end)])
            if frame_step != 1:
                cmd.extend(['--frame-step', str(frame_step)])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print(f"  ✓ 成功")
                if result.stdout:
                    print(f"  {result.stdout.strip()}")
                success_count += 1
            else:
                print(f"  ✗ 失败 (退出码: {result.returncode})")
                if result.stderr:
                    print(f"  错误: {result.stderr.strip()}")
                fail_count += 1
                
        except subprocess.TimeoutExpired:
            print(f"  ✗ 超时")
            fail_count += 1
        except Exception as e:
            print(f"  ✗ 错误: {e}")
            fail_count += 1
    
    print(f"\n完成!")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="批量从 Blender .blend 文件中导出相机参数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理目录中的所有 .blend 文件，输出到同目录
  python batch_export_cameras.py /home/alex/projects/FoundationGeo/data/blender
  
  # 指定输出目录
  python batch_export_cameras.py /home/alex/projects/FoundationGeo/data/blender -o output/
  
  # 指定相机名称和渲染尺寸
  python batch_export_cameras.py /home/alex/projects/FoundationGeo/data/blender -c Camera -w 1920 --height 1080
  
  # 只显示相机信息，不导出文件
  python batch_export_cameras.py /home/alex/projects/FoundationGeo/data/blender --only-camera
  
  # 导出动画中每一帧的相机参数
  python batch_export_cameras.py /home/alex/projects/FoundationGeo/data/blender --export-animation
        """
    )
    
    parser.add_argument("blend_dir", help="包含 .blend 文件的目录")
    parser.add_argument("-o", "--output", help="输出基础目录（默认：与 .blend 文件同目录）")
    parser.add_argument("-c", "--camera", help="相机名称（默认：活动相机）")
    parser.add_argument("-w", "--width", type=int, help="渲染宽度（默认：使用场景设置）")
    parser.add_argument("--height", type=int, help="渲染高度（默认：使用场景设置）")
    parser.add_argument("--only-camera", action="store_true",
                       help="只显示相机信息，不导出文件")
    parser.add_argument("--export-animation", action="store_true",
                       help="导出动画中每一帧的相机参数")
    parser.add_argument("--frame-start", type=int, default=None,
                       help="起始帧（默认：使用场景设置）")
    parser.add_argument("--frame-end", type=int, default=None,
                       help="结束帧（默认：使用场景设置）")
    parser.add_argument("--frame-step", type=int, default=1,
                       help="帧步长（默认：1，即每一帧）")
    parser.add_argument("--blender", help="Blender 可执行文件路径（默认：自动查找）")
    
    args = parser.parse_args()
    
    try:
        batch_export_cameras(
            args.blend_dir,
            args.output,
            args.camera,
            args.width,
            args.height,
            args.only_camera,
            args.export_animation,
            args.frame_start,
            args.frame_end,
            args.frame_step
        )
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
