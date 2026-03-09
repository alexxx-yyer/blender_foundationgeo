#!/usr/bin/env python3
"""Normalize filenames to {frame:06d}.{ext}."""

import argparse
import os
import re
import sys
from pathlib import Path


def extract_frame_number(filename, pattern):
    """Extract frame number from filename."""
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    return None


def rename_pose_files(pose_dir, dry_run=False):
    """Rename pose files: pose_000001.txt -> 000001.txt."""
    pose_dir = Path(pose_dir)
    if not pose_dir.exists():
        print(f"Warning: directory does not exist: {pose_dir}")
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
            print(f"Warning: target file exists, skipping: {file.name} -> {new_name}")
            continue
        
        if dry_run:
            print(f"  [Preview] {file.name} -> {new_name}")
        else:
            file.rename(new_path)
            print(f"  ✓ {file.name} -> {new_name}")
        renamed += 1
    
    return renamed


def rename_focal_files(focal_dir, dry_run=False):
    """Rename focal files: focal_000001.txt -> 000001.txt."""
    focal_dir = Path(focal_dir)
    if not focal_dir.exists():
        print(f"Warning: directory does not exist: {focal_dir}")
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
            print(f"Warning: target file exists, skipping: {file.name} -> {new_name}")
            continue
        
        if dry_run:
            print(f"  [Preview] {file.name} -> {new_name}")
        else:
            file.rename(new_path)
            print(f"  ✓ {file.name} -> {new_name}")
        renamed += 1
    
    return renamed


def rename_depth_files(depth_npy_dir, dry_run=False):
    """Rename depth files: depth0001.npy -> 000001.npy."""
    depth_npy_dir = Path(depth_npy_dir)
    if not depth_npy_dir.exists():
        print(f"Warning: directory does not exist: {depth_npy_dir}")
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
            print(f"Warning: target file exists, skipping: {file.name} -> {new_name}")
            continue
        
        if dry_run:
            print(f"  [Preview] {file.name} -> {new_name}")
        else:
            file.rename(new_path)
            print(f"  ✓ {file.name} -> {new_name}")
        renamed += 1
    
    return renamed


def rename_rgb_files(rgb_dir, dry_run=False):
    """Rename RGB files: glareOutput0001.png -> 000001.png."""
    rgb_dir = Path(rgb_dir)
    if not rgb_dir.exists():
        print(f"Warning: directory does not exist: {rgb_dir}")
        return 0
    
    renamed = 0
    # Support multiple possible naming patterns.
    patterns = [
        r'glareOutput(\d+)\.png',
        r'(\d+)\.png',  # If already normalized, skip below.
    ]
    
    for file in sorted(rgb_dir.glob('*.png')):
        # Skip if already normalized.
        if re.match(r'^\d{6}\.png$', file.name):
            continue
        
        frame_num = None
        for pattern in patterns:
            frame_num = extract_frame_number(file.name, pattern)
            if frame_num is not None:
                break
        
        if frame_num is None:
            print(f"Warning: cannot parse filename: {file.name}")
            continue
        
        new_name = f"{frame_num:06d}.png"
        new_path = rgb_dir / new_name
        
        if new_path.exists() and new_path != file:
            print(f"Warning: target file exists, skipping: {file.name} -> {new_name}")
            continue
        
        if dry_run:
            print(f"  [Preview] {file.name} -> {new_name}")
        else:
            file.rename(new_path)
            print(f"  ✓ {file.name} -> {new_name}")
        renamed += 1
    
    return renamed


def rename_all_files(data_dir, dry_run=False):
    """Normalize all supported files in the data directory."""
    data_dir = Path(data_dir).expanduser().resolve()
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {data_dir}")
    
    print(f"{'Preview mode' if dry_run else 'Execute mode'}: {data_dir}")
    print("=" * 60)
    
    total_renamed = 0
    
    # Rename pose files.
    pose_dir = data_dir / "pose"
    if pose_dir.exists():
        print(f"\nProcessing pose files ({pose_dir}):")
        count = rename_pose_files(pose_dir, dry_run)
        total_renamed += count
        print(f"  Processed {count} file(s)")
    
    # Rename focal files.
    focal_dir = data_dir / "focal"
    if focal_dir.exists():
        print(f"\nProcessing focal files ({focal_dir}):")
        count = rename_focal_files(focal_dir, dry_run)
        total_renamed += count
        print(f"  Processed {count} file(s)")
    
    # Rename depth files.
    depth_npy_dir = data_dir / "depth" / "npy"
    if depth_npy_dir.exists():
        print(f"\nProcessing depth files ({depth_npy_dir}):")
        count = rename_depth_files(depth_npy_dir, dry_run)
        total_renamed += count
        print(f"  Processed {count} file(s)")
    
    # Rename RGB files.
    rgb_dir = data_dir / "rgb"
    if rgb_dir.exists():
        print(f"\nProcessing RGB files ({rgb_dir}):")
        count = rename_rgb_files(rgb_dir, dry_run)
        total_renamed += count
        print(f"  Processed {count} file(s)")
    
    print("\n" + "=" * 60)
    if dry_run:
        print(f"Preview complete: {total_renamed} file(s) would be renamed")
        print("Use --execute to apply actual renames")
    else:
        print(f"Done: renamed {total_renamed} file(s)")
    
    return total_renamed


def main():
    parser = argparse.ArgumentParser(
        description="Normalize filenames to {frame:06d}.{ext}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview only (no actual rename)
  python rename_files.py /path/to/data --dry-run
  
  # Apply renaming
  python rename_files.py /path/to/data --execute
        """
    )
    
    parser.add_argument("data_dir", help="Path to data directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview mode: print planned renames only")
    parser.add_argument("--execute", action="store_true",
                        help="Execute mode: perform actual renaming")
    
    args = parser.parse_args()
    
    # If neither --dry-run nor --execute is provided, default to dry-run.
    if not args.dry_run and not args.execute:
        args.dry_run = True
        print("Note: defaulting to preview mode. Use --execute to apply changes.\n")
    
    try:
        rename_all_files(args.data_dir, dry_run=args.dry_run)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
