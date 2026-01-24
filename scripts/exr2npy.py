#!/usr/bin/env python3
"""将EXR文件转换为NPY格式"""

import os
import glob
import numpy as np
import OpenEXR
import Imath


def exr_to_npy(exr_path: str, npy_path: str | None = None) -> np.ndarray:
    """
    将EXR文件转换为NPY格式
    
    Args:
        exr_path: EXR文件路径
        npy_path: 输出的NPY文件路径，如果为None则使用与输入相同的文件名
    
    Returns:
        读取的numpy数组
    """
    exr_path = os.path.expanduser(exr_path)
    
    if not os.path.exists(exr_path):
        raise FileNotFoundError(f"找不到文件: {exr_path}")
    
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
    
    # 组合通道为图像数组
    if 'R' in channels and 'G' in channels and 'B' in channels:
        # RGB图像
        if 'A' in channels:
            img = np.stack([channel_data['R'], channel_data['G'], channel_data['B'], channel_data['A']], axis=-1)
        else:
            img = np.stack([channel_data['R'], channel_data['G'], channel_data['B']], axis=-1)
    elif 'Y' in channels:
        # 灰度图像
        img = channel_data['Y']
    else:
        # 按字母顺序堆叠所有通道
        sorted_channels = sorted(channels)
        if len(sorted_channels) == 1:
            img = channel_data[sorted_channels[0]]
        else:
            img = np.stack([channel_data[c] for c in sorted_channels], axis=-1)
    
    # 确定输出路径
    if npy_path is None:
        npy_path = os.path.splitext(exr_path)[0] + '.npy'
    else:
        npy_path = os.path.expanduser(npy_path)
    
    # 保存为NPY格式
    np.save(npy_path, img)
    
    print(f"转换完成!")
    print(f"  输入: {exr_path}")
    print(f"  输出: {npy_path}")
    print(f"  形状: {img.shape}")
    print(f"  数据类型: {img.dtype}")
    print(f"  通道: {channels}")
    
    return img


def batch_exr_to_npy(input_dir: str, output_dir: str | None = None, recursive: bool = False):
    """
    批量将目录中的EXR文件转换为NPY格式
    
    Args:
        input_dir: 包含EXR文件的目录
        output_dir: 输出目录，如果为None则使用输入目录
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
                npy_path = None
            else:
                # 保持相对路径结构
                rel_path = os.path.relpath(exr_file, input_dir)
                rel_dir = os.path.dirname(rel_path)
                base_name = os.path.splitext(os.path.basename(rel_path))[0]
                
                if rel_dir:
                    output_subdir = os.path.join(output_dir, rel_dir)
                    os.makedirs(output_subdir, exist_ok=True)
                    npy_path = os.path.join(output_subdir, base_name + '.npy')
                else:
                    npy_path = os.path.join(output_dir, base_name + '.npy')
            
            print(f"\n处理: {os.path.basename(exr_file)}")
            exr_to_npy(exr_file, npy_path)
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
        description="将EXR文件转换为NPY格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 转换单个文件:
  python exr2npy.py input.exr
  
  # 转换单个文件并指定输出:
  python exr2npy.py input.exr -o output.npy
  
  # 批量转换目录中的所有EXR文件:
  python exr2npy.py /path/to/exr/dir --batch
  
  # 批量转换并指定输出目录:
  python exr2npy.py /path/to/exr/dir --batch -o /path/to/output
  
  # 递归搜索子目录:
  python exr2npy.py /path/to/exr/dir --batch --recursive
        """
    )
    parser.add_argument("input", help="输入的EXR文件路径或目录")
    parser.add_argument("-o", "--output", help="输出的NPY文件路径或目录（可选）")
    parser.add_argument("--batch", action="store_true", 
                       help="批量处理模式：将输入视为目录并处理其中的所有EXR文件")
    parser.add_argument("-r", "--recursive", action="store_true",
                       help="递归搜索子目录（仅批量模式）")
    
    args = parser.parse_args()
    
    # 检查是文件还是目录
    input_path = os.path.expanduser(args.input)
    
    if args.batch or os.path.isdir(input_path):
        # 批量处理模式
        batch_exr_to_npy(input_path, args.output, args.recursive)
    else:
        # 单文件处理模式
        exr_to_npy(args.input, args.output)
