#!/usr/bin/env python3
"""统一文件命名格式为 {frame:06d}.{ext}"""

import argparse
import os
import re
import sys
from pathlib import Path


def extract_frame_number(filename, pattern):
    """从文件名中提取帧号"""
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    return None


def rename_pose_files(pose_dir, dry_run=False):
    """重命名 pose 文件: pose_000001.txt -> 000001.txt"""
    pose_dir = Path(pose_dir)
    if not pose_dir.exists():
        print(f"警告: 目录不存在: {pose_dir}")
        return 0
    
    renamed = 0
    pattern = r'pose_(\d+)\.txt'
    
    for file in sorted(pose_dir.glob('pose_*.txt')):
        frame_num = extract_frame_number(file.name, pattern)
        if frame_num is None:
            continue
        
        new_name = f"{frame_num:06d}.txt"
        new_path = pose_dir / new_name
        
        if new_path.exists() and new_path != file:
            print(f"警告: 目标文件已存在，跳过: {file.name} -> {new_name}")
            continue
        
        if dry_run:
            print(f"  [预览] {file.name} -> {new_name}")
        else:
            file.rename(new_path)
            print(f"  ✓ {file.name} -> {new_name}")
        renamed += 1
    
    return renamed


def rename_focal_files(focal_dir, dry_run=False):
    """重命名 focal 文件: focal_000001.txt -> 000001.txt"""
    focal_dir = Path(focal_dir)
    if not focal_dir.exists():
        print(f"警告: 目录不存在: {focal_dir}")
        return 0
    
    renamed = 0
    pattern = r'focal_(\d+)\.txt'
    
    for file in sorted(focal_dir.glob('focal_*.txt')):
        frame_num = extract_frame_number(file.name, pattern)
        if frame_num is None:
            continue
        
        new_name = f"{frame_num:06d}.txt"
        new_path = focal_dir / new_name
        
        if new_path.exists() and new_path != file:
            print(f"警告: 目标文件已存在，跳过: {file.name} -> {new_name}")
            continue
        
        if dry_run:
            print(f"  [预览] {file.name} -> {new_name}")
        else:
            file.rename(new_path)
            print(f"  ✓ {file.name} -> {new_name}")
        renamed += 1
    
    return renamed


def rename_depth_files(depth_npy_dir, dry_run=False):
    """重命名 depth 文件: depth0001.npy -> 000001.npy"""
    depth_npy_dir = Path(depth_npy_dir)
    if not depth_npy_dir.exists():
        print(f"警告: 目录不存在: {depth_npy_dir}")
        return 0
    
    renamed = 0
    pattern = r'depth(\d+)\.npy'
    
    for file in sorted(depth_npy_dir.glob('depth*.npy')):
        frame_num = extract_frame_number(file.name, pattern)
        if frame_num is None:
            continue
        
        new_name = f"{frame_num:06d}.npy"
        new_path = depth_npy_dir / new_name
        
        if new_path.exists() and new_path != file:
            print(f"警告: 目标文件已存在，跳过: {file.name} -> {new_name}")
            continue
        
        if dry_run:
            print(f"  [预览] {file.name} -> {new_name}")
        else:
            file.rename(new_path)
            print(f"  ✓ {file.name} -> {new_name}")
        renamed += 1
    
    return renamed


def rename_rgb_files(rgb_dir, dry_run=False):
    """重命名 RGB 文件: glareOutput0001.png -> 000001.png"""
    rgb_dir = Path(rgb_dir)
    if not rgb_dir.exists():
        print(f"警告: 目录不存在: {rgb_dir}")
        return 0
    
    renamed = 0
    # 支持多种可能的命名格式
    patterns = [
        r'glareOutput(\d+)\.png',
        r'(\d+)\.png',  # 如果已经是正确格式，跳过
    ]
    
    for file in sorted(rgb_dir.glob('*.png')):
        # 如果已经是正确格式，跳过
        if re.match(r'^\d{6}\.png$', file.name):
            continue
        
        frame_num = None
        for pattern in patterns:
            frame_num = extract_frame_number(file.name, pattern)
            if frame_num is not None:
                break
        
        if frame_num is None:
            print(f"警告: 无法解析文件名: {file.name}")
            continue
        
        new_name = f"{frame_num:06d}.png"
        new_path = rgb_dir / new_name
        
        if new_path.exists() and new_path != file:
            print(f"警告: 目标文件已存在，跳过: {file.name} -> {new_name}")
            continue
        
        if dry_run:
            print(f"  [预览] {file.name} -> {new_name}")
        else:
            file.rename(new_path)
            print(f"  ✓ {file.name} -> {new_name}")
        renamed += 1
    
    return renamed


def rename_all_files(data_dir, dry_run=False):
    """统一重命名所有文件"""
    data_dir = Path(data_dir).expanduser().resolve()
    
    if not data_dir.exists():
        raise FileNotFoundError(f"目录不存在: {data_dir}")
    
    print(f"{'预览模式' if dry_run else '执行模式'}: {data_dir}")
    print("=" * 60)
    
    total_renamed = 0
    
    # 重命名 pose 文件
    pose_dir = data_dir / "pose"
    if pose_dir.exists():
        print(f"\n处理 pose 文件 ({pose_dir}):")
        count = rename_pose_files(pose_dir, dry_run)
        total_renamed += count
        print(f"  共处理 {count} 个文件")
    
    # 重命名 focal 文件
    focal_dir = data_dir / "focal"
    if focal_dir.exists():
        print(f"\n处理 focal 文件 ({focal_dir}):")
        count = rename_focal_files(focal_dir, dry_run)
        total_renamed += count
        print(f"  共处理 {count} 个文件")
    
    # 重命名 depth 文件
    depth_npy_dir = data_dir / "depth" / "npy"
    if depth_npy_dir.exists():
        print(f"\n处理 depth 文件 ({depth_npy_dir}):")
        count = rename_depth_files(depth_npy_dir, dry_run)
        total_renamed += count
        print(f"  共处理 {count} 个文件")
    
    # 重命名 RGB 文件
    rgb_dir = data_dir / "rgb"
    if rgb_dir.exists():
        print(f"\n处理 RGB 文件 ({rgb_dir}):")
        count = rename_rgb_files(rgb_dir, dry_run)
        total_renamed += count
        print(f"  共处理 {count} 个文件")
    
    print("\n" + "=" * 60)
    if dry_run:
        print(f"预览完成: 将重命名 {total_renamed} 个文件")
        print("使用 --execute 参数执行实际重命名")
    else:
        print(f"完成: 已重命名 {total_renamed} 个文件")
    
    return total_renamed


def main():
    parser = argparse.ArgumentParser(
        description="统一文件命名格式为 {frame:06d}.{ext}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 预览重命名（不实际执行）
  python rename_files.py /path/to/data --dry-run
  
  # 执行重命名
  python rename_files.py /path/to/data --execute
        """
    )
    
    parser.add_argument("data_dir", help="数据目录路径")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览模式：只显示将要重命名的文件，不实际执行")
    parser.add_argument("--execute", action="store_true",
                        help="执行模式：实际执行重命名操作")
    
    args = parser.parse_args()
    
    # 如果没有指定 --dry-run 或 --execute，默认使用 --dry-run
    if not args.dry_run and not args.execute:
        args.dry_run = True
        print("注意: 默认使用预览模式，使用 --execute 执行实际重命名\n")
    
    try:
        rename_all_files(args.data_dir, dry_run=args.dry_run)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
