#!/usr/bin/env python3
"""将EXR深度文件转换为伪彩色PNG"""

import os
import glob
import numpy as np
import OpenEXR
import Imath
import matplotlib.pyplot as plt


def exr_to_png(exr_path: str, png_path: str | None = None, colormap: str = 'viridis', 
               vmin: float | None = None, vmax: float | None = None, invert: bool = False) -> np.ndarray:
    """
    将EXR深度文件转换为伪彩色PNG
    
    Args:
        exr_path: EXR文件路径
        png_path: 输出的PNG文件路径，如果为None则使用与输入相同的文件名
        colormap: matplotlib colormap名称 (viridis, jet, turbo, plasma, inferno等)
        vmin: 深度最小值，None则自动
        vmax: 深度最大值，None则自动
    
    Returns:
        读取的numpy数组
    """
    exr_path = os.path.expanduser(exr_path)
    
    if not os.path.exists(exr_path):
        raise FileNotFoundError(f"File not found: {exr_path}")
    
    # 使用OpenEXR读取文件
    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()
    
    # 获取图像尺寸
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # 获取通道列表
    channels = list(header['channels'].keys())
    
    # 确定像素类型
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    
    # 读取所有通道
    channel_data = {}
    for channel in channels:
        raw_data = exr_file.channel(channel, pt)
        channel_data[channel] = np.frombuffer(raw_data, dtype=np.float32).reshape(height, width)
    
    # 获取深度数据（取第一个通道或特定通道）
    if len(channels) == 1:
        depth = channel_data[channels[0]]
    elif 'Z' in channels:
        depth = channel_data['Z']
    elif 'depth' in channels:
        depth = channel_data['depth']
    else:
        depth = channel_data[sorted(channels)[0]]
    
    # 归一化深度值
    if vmin is None:
        vmin = np.min(depth)
    if vmax is None:
        vmax = np.max(depth)
    
    depth_normalized = (depth - vmin) / (vmax - vmin + 1e-8)
    depth_normalized = np.clip(depth_normalized, 0, 1)
    
    # 反转深度值（近=红，远=蓝）
    if invert:
        depth_normalized = 1.0 - depth_normalized
    
    # 应用伪彩色
    cmap = plt.colormaps[colormap]
    colored = cmap(depth_normalized)
    
    # 转换为8位RGB
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    
    # 确定输出路径
    if png_path is None:
        png_path = os.path.splitext(exr_path)[0] + '.png'
    else:
        png_path = os.path.expanduser(png_path)
    
    # 保存PNG
    plt.imsave(png_path, rgb)
    
    print(f"Done!")
    print(f"  Input: {exr_path}")
    print(f"  Output: {png_path}")
    print(f"  Shape: {depth.shape}")
    print(f"  Depth range: [{vmin:.4f}, {vmax:.4f}]")
    print(f"  Colormap: {colormap}")
    print(f"  Inverted: {invert}")
    print(f"  Channels: {channels}")
    
    return depth


def batch_exr_to_png(input_dir: str, output_dir: str | None = None, 
                     colormap: str = 'turbo', vmin: float | None = None, 
                     vmax: float | None = None, invert: bool = False,
                     recursive: bool = False):
    """
    批量将目录中的EXR文件转换为PNG格式
    
    Args:
        input_dir: 包含EXR文件的目录
        output_dir: 输出目录，如果为None则使用输入目录
        colormap: matplotlib colormap名称
        vmin: 深度最小值，None则自动
        vmax: 深度最大值，None则自动
        invert: 是否反转深度值
        recursive: 是否递归搜索子目录
    """
    input_dir = os.path.expanduser(input_dir)
    
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"目录不存在: {input_dir}")
    
    # 查找所有EXR文件
    if recursive:
        pattern = os.path.join(input_dir, "**", "*.exr")
        exr_files = glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(input_dir, "*.exr")
        exr_files = glob.glob(pattern)
    
    if not exr_files:
        print(f"在目录 {input_dir} 中未找到EXR文件")
        return
    
    print(f"找到 {len(exr_files)} 个EXR文件")
    
    # 确定输出目录
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = os.path.expanduser(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个文件
    success_count = 0
    fail_count = 0
    
    for exr_file in sorted(exr_files):
        try:
            # 确定输出路径
            if output_dir == input_dir:
                # 输出到同目录
                png_path = None
            else:
                # 保持相对路径结构
                rel_path = os.path.relpath(exr_file, input_dir)
                rel_dir = os.path.dirname(rel_path)
                base_name = os.path.splitext(os.path.basename(rel_path))[0]
                
                if rel_dir:
                    output_subdir = os.path.join(output_dir, rel_dir)
                    os.makedirs(output_subdir, exist_ok=True)
                    png_path = os.path.join(output_subdir, base_name + '.png')
                else:
                    png_path = os.path.join(output_dir, base_name + '.png')
            
            print(f"\n处理: {os.path.basename(exr_file)}")
            exr_to_png(exr_file, png_path, colormap, vmin, vmax, invert)
            success_count += 1
        except Exception as e:
            print(f"  ✗ 错误: {e}")
            fail_count += 1
    
    print(f"\n批量转换完成!")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert EXR depth to pseudo-color PNG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 转换单个文件:
  python exr2png.py input.exr
  
  # 转换单个文件并指定输出:
  python exr2png.py input.exr -o output.png
  
  # 批量转换目录中的所有EXR文件:
  python exr2png.py /path/to/exr/dir --batch
  
  # 批量转换并指定输出目录:
  python exr2png.py /path/to/exr/dir --batch -o /path/to/output
  
  # 递归搜索子目录:
  python exr2png.py /path/to/exr/dir --batch --recursive
        """
    )
    parser.add_argument("input", help="输入的EXR文件路径或目录")
    parser.add_argument("-o", "--output", help="输出的PNG文件路径或目录（可选）")
    parser.add_argument("-c", "--colormap", default="turbo", 
                        help="Colormap name: turbo, turbo_r, viridis, jet, plasma, inferno (default: turbo)")
    parser.add_argument("--vmin", type=float, help="Min depth value")
    parser.add_argument("--vmax", type=float, help="Max depth value")
    parser.add_argument("-i", "--invert", action="store_true", help="Invert depth (near=red, far=blue)")
    parser.add_argument("--batch", action="store_true", 
                       help="批量处理模式：将输入视为目录并处理其中的所有EXR文件")
    parser.add_argument("-r", "--recursive", action="store_true",
                       help="递归搜索子目录（仅批量模式）")
    
    args = parser.parse_args()
    
    # 检查是文件还是目录
    input_path = os.path.expanduser(args.input)
    
    if args.batch or os.path.isdir(input_path):
        # 批量处理模式
        batch_exr_to_png(input_path, args.output, args.colormap, 
                        args.vmin, args.vmax, args.invert, args.recursive)
    else:
        # 单文件处理模式
        exr_to_png(args.input, args.output, args.colormap, args.vmin, args.vmax, args.invert)
