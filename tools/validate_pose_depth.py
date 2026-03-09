#!/usr/bin/env python3
"""Validate pose and depth data quality"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def load_pose(pose_path: Path) -> Optional[np.ndarray]:
    """Load pose matrix (4x4)"""
    try:
        pose = np.loadtxt(pose_path)
        if pose.shape != (4, 4):
            return None
        return pose
    except Exception:
        return None


def load_focal(focal_path: Path) -> Optional[Tuple[float, float]]:
    """Load focal values (fx, fy)"""
    try:
        with open(focal_path, 'r') as f:
            line = f.readline().strip()
            values = line.split()
            if len(values) == 1:
                fx = fy = float(values[0])
            elif len(values) == 2:
                fx, fy = float(values[0]), float(values[1])
            else:
                return None
            return (fx, fy)
    except Exception:
        return None


def load_depth(depth_path: Path) -> Optional[np.ndarray]:
    """Load depth map"""
    try:
        depth = np.load(depth_path)
        return depth
    except Exception:
        return None


def validate_pose_format(pose: np.ndarray) -> Dict:
    """Validate pose format"""
    errors = []
    warnings = []
    
    # Check shape
    if pose.shape != (4, 4):
        errors.append(f"Invalid pose matrix shape: {pose.shape}, expected (4, 4)")
        return {"valid": False, "errors": errors, "warnings": warnings}
    
    # Check numeric validity
    if np.any(np.isnan(pose)) or np.any(np.isinf(pose)):
        errors.append("Pose matrix contains NaN or Inf")
    
    # Check last homogeneous row
    expected_last_row = np.array([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(pose[3, :], expected_last_row, atol=1e-6):
        warnings.append(f"Pose matrix last row is not homogeneous: {pose[3, :]}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


def validate_pose_geometry(pose: np.ndarray) -> Dict:
    """Validate pose geometry"""
    errors = []
    warnings = []
    
    # Extract rotation and translation
    R = pose[:3, :3]
    t = pose[:3, 3]
    
    # Check rotation orthogonality: R^T R = I
    RTR = R.T @ R
    I = np.eye(3)
    ortho_error = np.max(np.abs(RTR - I))
    if ortho_error > 1e-4:
        errors.append(f"Rotation matrix is not orthogonal: max error = {ortho_error:.6f}")
    elif ortho_error > 1e-6:
        warnings.append(f"Large rotation orthogonality error: {ortho_error:.6f}")
    
    # Check determinant: det(R) ≈ 1
    det_R = np.linalg.det(R)
    if abs(det_R - 1.0) > 1e-4:
        errors.append(f"Rotation determinant is not 1: det(R) = {det_R:.6f}")
    elif abs(det_R - 1.0) > 1e-6:
        warnings.append(f"Rotation determinant deviation: det(R) = {det_R:.6f}")
    
    # Check translation magnitude reasonableness
    t_norm = np.linalg.norm(t)
    if t_norm > 1e6:
        warnings.append(f"Abnormal translation magnitude: ||t|| = {t_norm:.2f}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "ortho_error": ortho_error,
        "det_R": det_R,
        "t_norm": t_norm
    }


def validate_pose_sequence(poses: List[np.ndarray]) -> Dict:
    """Validate pose sequence continuity"""
    errors = []
    warnings = []
    
    if len(poses) < 2:
        return {"valid": True, "errors": [], "warnings": []}
    
    # Compute relative transforms between adjacent frames
    relative_transforms = []
    position_changes = []
    rotation_changes = []
    
    for i in range(len(poses) - 1):
        pose_i = poses[i]
        pose_j = poses[i + 1]
        
        # Relative transform: T_rel = T_j @ T_i^{-1}
        pose_i_inv = np.linalg.inv(pose_i)
        T_rel = pose_j @ pose_i_inv
        
        relative_transforms.append(T_rel)
        
        # Extract positional change
        pos_i = pose_i[:3, 3]
        pos_j = pose_j[:3, 3]
        pos_change = np.linalg.norm(pos_j - pos_i)
        position_changes.append(pos_change)
        
        # Extract rotation change
        R_i = pose_i[:3, :3]
        R_j = pose_j[:3, :3]
        R_rel = T_rel[:3, :3]
        # Compute rotation angle from matrix
        trace = np.trace(R_rel)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        rotation_changes.append(angle)
    
    # Detect abnormal jumps
    if position_changes:
        pos_mean = np.mean(position_changes)
        pos_std = np.std(position_changes)
        pos_threshold = pos_mean + 3 * pos_std
        
        for i, pos_change in enumerate(position_changes):
            if pos_change > pos_threshold:
                warnings.append(f"Frame {i+1}->{i+2} Abnormal position change: {pos_change:.4f} (mean: {pos_mean:.4f}, threshold: {pos_threshold:.4f})")
    
    if rotation_changes:
        rot_mean = np.mean(rotation_changes)
        rot_std = np.std(rotation_changes)
        rot_threshold = rot_mean + 3 * rot_std
        
        for i, rot_change in enumerate(rotation_changes):
            if rot_change > rot_threshold:
                warnings.append(f"Frame {i+1}->{i+2} Abnormal rotation change: {np.degrees(rot_change):.2f}° (mean: {np.degrees(rot_mean):.2f}°, threshold: {np.degrees(rot_threshold):.2f}°)")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "position_changes": position_changes,
        "rotation_changes": rotation_changes,
        "mean_position_change": np.mean(position_changes) if position_changes else 0,
        "mean_rotation_change": np.mean(rotation_changes) if rotation_changes else 0
    }


def validate_depth_numerical(depth: np.ndarray) -> Dict:
    """Validate depth numerics"""
    errors = []
    warnings = []
    stats = {}
    
    # Check numeric validity
    if np.any(np.isnan(depth)):
        nan_count = np.sum(np.isnan(depth))
        errors.append(f"Depth map contains {nan_count} NaN value(s)")
    
    if np.any(np.isinf(depth)):
        inf_count = np.sum(np.isinf(depth))
        errors.append(f"Depth map contains {inf_count} Inf value(s)")
    
    # Check non-negativity
    negative_count = np.sum(depth < 0)
    if negative_count > 0:
        errors.append(f"Depth map contains {negative_count} negative value(s)")
    
    # Statistics
    valid_depth = depth[np.isfinite(depth) & (depth >= 0)]
    if len(valid_depth) > 0:
        stats["min"] = float(np.min(valid_depth))
        stats["max"] = float(np.max(valid_depth))
        stats["mean"] = float(np.mean(valid_depth))
        stats["std"] = float(np.std(valid_depth))
        stats["valid_pixels"] = int(len(valid_depth))
        stats["total_pixels"] = int(depth.size)
        stats["invalid_ratio"] = float(1.0 - len(valid_depth) / depth.size)
        
        # Check depth range reasonableness
        if stats["max"] > 1e6:
            # Check outlier ratio
            large_depth_count = np.sum(valid_depth > 1e6)
            large_depth_ratio = large_depth_count / len(valid_depth) if len(valid_depth) > 0 else 0
            if large_depth_ratio > 0.01:  # if more than 1% pixels are outliers
                warnings.append(f"Abnormal depth max: {stats['max']:.2e} ({large_depth_ratio*100:.2f}% pixels > 1e6)")
            elif large_depth_count > 0:
                # if only a few outliers, keep as informational warning
                warnings.append(f"Abnormal depth max: {stats['max']:.2e} (only {large_depth_count} pixels, likely noise or render artifact)")
        if stats["min"] < 0.01:
            warnings.append(f"Depth minimum is too small: {stats['min']:.6f}")
    else:
        errors.append("Depth map has no valid pixels")
        stats = {}
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stats": stats
    }


def validate_depth_geometry(depth: np.ndarray) -> Dict:
    """Validate depth geometry"""
    errors = []
    warnings = []
    
    # Compute depth gradients
    grad_x = np.gradient(depth, axis=1)
    grad_y = np.gradient(depth, axis=0)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Check depth gradient reasonableness
    valid_mask = np.isfinite(depth) & (depth > 0)
    if np.any(valid_mask):
        valid_grad = grad_magnitude[valid_mask]
        max_grad = np.max(valid_grad)
        mean_grad = np.mean(valid_grad)
        
        # Detect unusually large gradients
        # Note: object edges and occlusion boundaries can naturally have large gradients
        grad_threshold = mean_grad + 5 * np.std(valid_grad)
        large_grad_count = np.sum(valid_grad > grad_threshold)
        large_grad_ratio = large_grad_count / len(valid_grad) if len(valid_grad) > 0 else 0
        
        # Warn only when abnormal gradient ratio is high (>10%)
        if large_grad_ratio > 0.1:
            warnings.append(f"Detected {large_grad_count} abnormally large depth gradients ({large_grad_ratio*100:.1f}%, threshold: {grad_threshold:.4f}) - may correspond to edges or occlusions")
        elif large_grad_count > 0:
            # if ratio is low, keep informational only
            pass
    else:
        errors.append("Cannot compute depth gradients: no valid pixels")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "max_gradient": float(np.max(grad_magnitude)) if np.any(valid_mask) else None,
        "mean_gradient": float(np.mean(valid_grad)) if np.any(valid_mask) else None
    }


def validate_geometry_consistency(poses: List[np.ndarray], focals: List[Tuple[float, float]],
                                  depths: List[np.ndarray], width: int, height: int) -> Dict:
    """Validate geometric consistency (3D reconstruction and projection)"""
    errors = []
    warnings = []
    
    if len(poses) == 0 or len(focals) == 0 or len(depths) == 0:
        errors.append("Missing required pose/focal/depth data")
        return {"valid": False, "errors": errors, "warnings": warnings}
    
    if len(poses) != len(focals) or len(poses) != len(depths):
        errors.append(f"Data count mismatch: poses={len(poses)}, focals={len(focals)}, depths={len(depths)}")
        return {"valid": False, "errors": errors, "warnings": warnings}
    
    # Select key pixels for validation (image center and four corners).
    sample_points = [
        (width // 2, height // 2),  # Center
        (width // 4, height // 4),  # Top-left
        (3 * width // 4, height // 4),  # Top-right
        (width // 4, 3 * height // 4),  # Bottom-left
        (3 * width // 4, 3 * height // 4),  # Bottom-right
    ]
    
    # Reconstruct 3D points per view and transform to world space
    world_points_all = []
    
    for frame_idx, (pose, focal, depth) in enumerate(zip(poses, focals, depths)):
        fx, fy = focal
        cx, cy = width / 2, height / 2  # Assume principal point at image center.
        
        # Reconstruct 3D points in camera coordinates
        camera_points = []
        for u, v in sample_points:
            if 0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]:
                d = depth[v, u]
                if np.isfinite(d) and d > 0:
                    x = (u - cx) * d / fx
                    y = (v - cy) * d / fy
                    z = d
                    camera_points.append(np.array([x, y, z, 1.0]))
        
        # Transform to world coordinates
        for pt_cam in camera_points:
            pt_world = np.linalg.inv(pose) @ pt_cam
            world_points_all.append((frame_idx, pt_world[:3]))
    
    # Check multi-view consistency (simplified with adjacent frame proximity).
    if len(world_points_all) > 0:
        # Compute spatial distribution of reconstructed points
        points = np.array([pt[1] for pt in world_points_all])
        if len(points) > 1:
            point_distances = []
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    dist = np.linalg.norm(points[i] - points[j])
                    point_distances.append(dist)
            
            if point_distances:
                mean_dist = np.mean(point_distances)
                if mean_dist > 100:  # Tunable threshold.
                    warnings.append(f"Reconstructed 3D points are abnormally scattered: mean distance = {mean_dist:.2f}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "num_world_points": len(world_points_all)
    }


def visualize_pose_trajectory(poses: List[np.ndarray], output_path: Optional[Path] = None):
    """Visualize camera trajectory"""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping visualization")
        return
    
    positions = np.array([pose[:3, 3] for pose in poses])
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Camera Trajectory')
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='green', s=100, marker='o', label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='red', s=100, marker='s', label='End')
    
    # Draw camera orientation (every few frames)
    step = max(1, len(poses) // 10)
    for i in range(0, len(poses), step):
        pose = poses[i]
        pos = pose[:3, 3]
        # Camera forward direction (negative Z axis)
        forward = -pose[:3, 2]
        ax.quiver(pos[0], pos[1], pos[2], forward[0], forward[1], forward[2],
                  length=0.5, color='red', arrow_length_ratio=0.3)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Trajectory')
    ax.legend()
    ax.grid(True)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved trajectory plot: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_depth_stats(depths: List[np.ndarray], output_path: Optional[Path] = None):
    """Visualize depth statistics"""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping visualization")
        return
    
    # Collect all valid depth values using numpy (much faster than Python list)
    valid_depths_list = []
    frame_means = []
    frame_stds = []
    
    for depth in depths:
        valid_mask = np.isfinite(depth) & (depth > 0)
        valid = depth[valid_mask]
        if len(valid) > 0:
            valid_depths_list.append(valid)
            frame_means.append(np.mean(valid))
            frame_stds.append(np.std(valid))
        else:
            frame_means.append(0)
            frame_stds.append(0)
    
    if not valid_depths_list:
        print("Warning: No valid depth data for visualization")
        return
    
    # Concatenate all valid depths into a single numpy array
    all_depths = np.concatenate(valid_depths_list)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Depth histogram - use numpy histogram for better performance
    counts, bins = np.histogram(all_depths, bins=50)
    axes[0, 0].hist(bins[:-1], bins=bins, weights=counts, edgecolor='black')
    axes[0, 0].set_xlabel('Depth Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Depth Value Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(frame_means, label='Mean')
    axes[0, 1].fill_between(range(len(frame_means)),
                            np.array(frame_means) - np.array(frame_stds),
                            np.array(frame_means) + np.array(frame_stds),
                            alpha=0.3, label='±1 Std Dev')
    axes[0, 1].set_xlabel('Frame Number')
    axes[0, 1].set_ylabel('Depth Value')
    axes[0, 1].set_title('Depth Statistics per Frame')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Example depth maps (first frame and middle frame)
    # Downsample large images for faster rendering
    if len(depths) > 0:
        depth0 = depths[0]
        # Downsample if image is too large (max 1000 pixels on longest side)
        max_size = 1000
        if depth0.shape[0] > max_size or depth0.shape[1] > max_size:
            scale = max_size / max(depth0.shape[0], depth0.shape[1])
            new_h, new_w = int(depth0.shape[0] * scale), int(depth0.shape[1] * scale)
            # Simple downsampling using array indexing
            step_h, step_w = depth0.shape[0] / new_h, depth0.shape[1] / new_w
            indices_h = (np.arange(new_h) * step_h).astype(int)
            indices_w = (np.arange(new_w) * step_w).astype(int)
            depth0_ds = depth0[np.ix_(indices_h, indices_w)]
        else:
            depth0_ds = depth0
        
        im0 = axes[1, 0].imshow(depth0_ds, cmap='turbo', interpolation='nearest')
        axes[1, 0].set_title(f'Frame 1 Depth Map')
        axes[1, 0].axis('off')
        plt.colorbar(im0, ax=axes[1, 0])
        
        mid_idx = len(depths) // 2
        depth_mid = depths[mid_idx]
        if depth_mid.shape[0] > max_size or depth_mid.shape[1] > max_size:
            scale = max_size / max(depth_mid.shape[0], depth_mid.shape[1])
            new_h, new_w = int(depth_mid.shape[0] * scale), int(depth_mid.shape[1] * scale)
            # Simple downsampling using array indexing
            step_h, step_w = depth_mid.shape[0] / new_h, depth_mid.shape[1] / new_w
            indices_h = (np.arange(new_h) * step_h).astype(int)
            indices_w = (np.arange(new_w) * step_w).astype(int)
            depth_mid_ds = depth_mid[np.ix_(indices_h, indices_w)]
        else:
            depth_mid_ds = depth_mid
        
        im1 = axes[1, 1].imshow(depth_mid_ds, cmap='turbo', interpolation='nearest')
        axes[1, 1].set_title(f'Frame {mid_idx+1} Depth Map')
        axes[1, 1].axis('off')
        plt.colorbar(im1, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved depth statistics plot: {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_pointcloud(poses: List[np.ndarray], focals: List[Tuple[float, float]],
                       depths: List[np.ndarray], rgb_images: Optional[List[np.ndarray]] = None,
                       width: Optional[int] = None, height: Optional[int] = None,
                       downsample: int = 1, max_points: Optional[int] = None,
                       use_pose_direct: bool = False,
                       depth_min: Optional[float] = None, depth_max: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Generate point cloud from depth maps and camera poses
    
    Args:
        poses: List of camera pose matrices (4x4)
        focals: List of (fx, fy) focal lengths
        depths: List of depth maps
        rgb_images: Optional list of RGB images for coloring points
        width: Image width (if None, inferred from depth)
        height: Image height (if None, inferred from depth)
        downsample: Downsampling factor (1 = no downsampling, 2 = every 2nd pixel, etc.)
        max_points: Maximum number of points to include (None = all)
        use_pose_direct: If True, treat pose as camera-to-world (direct), else world-to-camera (inverse)
        depth_min: Minimum depth value (camera Z) to include (None = no limit)
        depth_max: Maximum depth value (camera Z) to include (None = no limit)
    
    Returns:
        points: Nx3 array of 3D points in world coordinates
        colors: Nx3 array of RGB colors (if rgb_images provided), None otherwise
    """
    if len(poses) == 0 or len(focals) == 0 or len(depths) == 0:
        return np.array([]).reshape(0, 3), None
    
    if len(poses) != len(focals) or len(poses) != len(depths):
        raise ValueError(f"Data count mismatch: poses={len(poses)}, focals={len(focals)}, depths={len(depths)}")
    
    all_points = []
    all_colors = [] if rgb_images is not None else None
    
    for frame_idx, (pose, focal, depth) in enumerate(zip(poses, focals, depths)):
        fx, fy = focal
        
        # Get image dimensions
        if width is None or height is None:
            height, width = depth.shape[:2]
        else:
            # Ensure depth matches expected size
            if depth.shape[0] != height or depth.shape[1] != width:
                depth = depth[:height, :width]
        
        cx, cy = width / 2, height / 2  # Assume principal point at image center
        
        # Downsample coordinates
        u_coords = np.arange(0, width, downsample)
        v_coords = np.arange(0, height, downsample)
        u_grid, v_grid = np.meshgrid(u_coords, v_coords)
        
        # Get depth values at sampled coordinates
        d = depth[v_grid, u_grid]
        
        # Filter valid depths
        valid_mask = np.isfinite(d) & (d > 0)
        
        # Apply depth range filter if specified
        if depth_min is not None:
            valid_mask = valid_mask & (d >= depth_min)
        if depth_max is not None:
            valid_mask = valid_mask & (d <= depth_max)
        
        if not np.any(valid_mask):
            continue
        
        u_valid = u_grid[valid_mask]
        v_valid = v_grid[valid_mask]
        d_valid = d[valid_mask]
        
        # Reconstruct 3D points in camera coordinates
        x_cam = (u_valid - cx) * d_valid / fx
        y_cam = (v_valid - cy) * d_valid / fy
        z_cam = d_valid
        
        # Stack into homogeneous coordinates
        points_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=1)
        
        # Transform to world coordinates
        # Note: According to export_camera.py, pose is "world-to-camera" transformation
        # However, Blender's matrix_world is actually "object-to-world", so we need to check
        # If use_pose_direct is True, treat pose as camera-to-world (direct use)
        # Otherwise, treat pose as world-to-camera (use inverse)
        if use_pose_direct:
            # Direct transformation: P_world = pose @ P_cam
            points_world = (pose @ points_cam.T).T[:, :3]
        else:
            # Inverse transformation: P_world = inv(pose) @ P_cam
            pose_inv = np.linalg.inv(pose)
            points_world = (pose_inv @ points_cam.T).T[:, :3]
        
        # Note: Depth filtering is already done above (depth_min, depth_max)
        # We don't filter by 3D world coordinates here - only by depth (Z in camera space)
        
        # Debug: check first frame point ranges
        if frame_idx == 0 and len(points_world) > 0:
            print(f"  Frame {frame_idx} debug: {len(points_world)} points")
            print(f"    Camera coords range: X[{x_cam.min():.2f}, {x_cam.max():.2f}], "
                  f"Y[{y_cam.min():.2f}, {y_cam.max():.2f}], Z[{z_cam.min():.2f}, {z_cam.max():.2f}]")
            print(f"    World coords range: X[{points_world[:, 0].min():.2f}, {points_world[:, 0].max():.2f}], "
                  f"Y[{points_world[:, 1].min():.2f}, {points_world[:, 1].max():.2f}], "
                  f"Z[{points_world[:, 2].min():.2f}, {points_world[:, 2].max():.2f}]")
        
        all_points.append(points_world)
        
        # Get colors if RGB images provided
        if rgb_images is not None and frame_idx < len(rgb_images):
            rgb = rgb_images[frame_idx]
            if rgb is not None:
                # Ensure RGB matches depth size
                if rgb.shape[0] != height or rgb.shape[1] != width:
                    rgb = rgb[:height, :width]
                
                # Get colors at valid points
                colors_frame = rgb[v_valid, u_valid]
                if colors_frame.ndim == 1:  # Grayscale (single point)
                    colors_frame = np.stack([colors_frame, colors_frame, colors_frame], axis=0)
                elif colors_frame.ndim == 2:
                    if colors_frame.shape[1] == 1:  # Grayscale (N, 1)
                        colors_frame = np.repeat(colors_frame, 3, axis=1)
                    elif colors_frame.shape[1] == 4:  # RGBA (N, 4)
                        colors_frame = colors_frame[:, :3]  # Drop alpha
                
                # Normalize to [0, 1] if needed
                if colors_frame.dtype == np.uint8:
                    colors_frame = colors_frame.astype(np.float32) / 255.0
                
                if all_colors is None:
                    all_colors = []
                all_colors.append(colors_frame)
    
    if not all_points:
        return np.array([]).reshape(0, 3), None
    
    # Concatenate all points
    points = np.concatenate(all_points, axis=0)
    
    # Concatenate colors if available
    colors = None
    if all_colors is not None:
        colors = np.concatenate(all_colors, axis=0)
    
    # Limit number of points if requested
    if max_points is not None and len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        if colors is not None:
            colors = colors[indices]
    
    return points, colors


def save_pointcloud_ply(points: np.ndarray, output_path: Path, colors: Optional[np.ndarray] = None):
    """
    Save point cloud to PLY format
    
    Args:
        points: Nx3 array of 3D points
        output_path: Output file path
        colors: Optional Nx3 array of RGB colors (0-1 range)
    """
    if len(points) == 0:
        raise ValueError("Cannot save empty point cloud")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert colors to uint8 if provided
    if colors is not None:
        if colors.max() <= 1.0:
            colors_uint8 = (colors * 255).astype(np.uint8)
        else:
            colors_uint8 = colors.astype(np.uint8)
    else:
        colors_uint8 = None
    
    with open(output_path, 'wb') as f:
        # Write PLY header
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {len(points)}\n".encode())
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        if colors_uint8 is not None:
            f.write(b"property uchar red\n")
            f.write(b"property uchar green\n")
            f.write(b"property uchar blue\n")
        f.write(b"end_header\n")
        
        # Write points (ensure correct byte order and format)
        points_float32 = points.astype(np.float32)
        if colors_uint8 is not None:
            for i in range(len(points)):
                # Write x, y, z as float32
                f.write(points_float32[i].tobytes())
                # Write r, g, b as uint8
                f.write(colors_uint8[i].tobytes())
        else:
            # Write all points at once (more efficient)
            f.write(points_float32.tobytes())
    
    print(f"  Saved point cloud: {output_path} ({len(points):,} points)")


def validate_all(data_dir: Path, pose_only: bool = False, depth_only: bool = False,
                 visualize: bool = False, output_dir: Optional[Path] = None,
                 export_pointcloud: bool = False, pointcloud_downsample: int = 2,
                 pointcloud_max_points: Optional[int] = None,
                 pointcloud_depth_min: Optional[float] = None,
                 pointcloud_depth_max: Optional[float] = None) -> Dict:
    """Run full validation"""
    data_dir = Path(data_dir).expanduser().resolve()
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {data_dir}")
    
    results = {
        "data_dir": str(data_dir),
        "pose_validation": {},
        "depth_validation": {},
        "geometry_validation": {},
        "summary": {}
    }
    
    # Find all files
    pose_dir = data_dir / "pose"
    focal_dir = data_dir / "focal"
    depth_dir = data_dir / "depth" / "npy"
    rgb_dir = data_dir / "rgb"
    
    # Collect all frame indices.
    if pose_dir.exists():
        pose_files = sorted(pose_dir.glob("*.txt"))
        frame_numbers = [int(f.stem) for f in pose_files]
    else:
        frame_numbers = []
    
    print(f"Found {len(frame_numbers)} frame(s) of data")
    print("=" * 60)
    
    # Validate poses.
    if not depth_only:
        print("\nValidating poses...")
        poses = []
        focals = []
        pose_format_results = []
        pose_geometry_results = []
        
        for frame_num in frame_numbers:
            pose_path = pose_dir / f"{frame_num:06d}.txt"
            focal_path = focal_dir / f"{frame_num:06d}.txt"
            
            if not pose_path.exists():
                print(f"  Warning: pose file missing: {pose_path.name}")
                continue
            
            pose = load_pose(pose_path)
            if pose is None:
                print(f"  Error: failed to load pose: {pose_path.name}")
                continue
            
            # Format validation
            format_result = validate_pose_format(pose)
            pose_format_results.append((frame_num, format_result))
            
            # Geometry validation
            geometry_result = validate_pose_geometry(pose)
            pose_geometry_results.append((frame_num, geometry_result))
            
            poses.append(pose)
            
            # Load focal values
            if focal_path.exists():
                focal = load_focal(focal_path)
                if focal:
                    focals.append(focal)
        
        # Sequence continuity validation
        sequence_result = validate_pose_sequence(poses) if len(poses) > 1 else {}
        
        results["pose_validation"] = {
            "format": pose_format_results,
            "geometry": pose_geometry_results,
            "sequence": sequence_result,
            "total_frames": len(poses)
        }
        
        # Count errors and warnings
        total_errors = sum(len(r[1]["errors"]) for r in pose_format_results + pose_geometry_results)
        total_warnings = sum(len(r[1]["warnings"]) for r in pose_format_results + pose_geometry_results)
        if sequence_result:
            total_errors += len(sequence_result.get("errors", []))
            total_warnings += len(sequence_result.get("warnings", []))
        
        print(f"  Pose validation complete: {len(poses)} frame(s), {total_errors} error(s), {total_warnings} warning(s)")
    
    # Validate depth.
    if not pose_only:
        print("\nValidating depth...")
        depths = []
        depth_numerical_results = []
        depth_geometry_results = []
        
        for frame_num in frame_numbers:
            depth_path = depth_dir / f"{frame_num:06d}.npy"
            
            if not depth_path.exists():
                print(f"  Warning: depth file missing: {depth_path.name}")
                continue
            
            depth = load_depth(depth_path)
            if depth is None:
                print(f"  Error: failed to load depth: {depth_path.name}")
                continue
            
            # Numeric validation
            numerical_result = validate_depth_numerical(depth)
            depth_numerical_results.append((frame_num, numerical_result))
            
            # Print numeric validation warnings.
            if numerical_result.get("warnings"):
                for warning in numerical_result["warnings"]:
                    print(f"  Warning [frame {frame_num:06d} numeric]: {warning}")
            
            # Geometry validation
            geometry_result = validate_depth_geometry(depth)
            depth_geometry_results.append((frame_num, geometry_result))
            
            # Print geometry validation warnings.
            if geometry_result.get("warnings"):
                for warning in geometry_result["warnings"]:
                    print(f"  Warning [frame {frame_num:06d} geometry]: {warning}")
            
            depths.append(depth)
        
        results["depth_validation"] = {
            "numerical": depth_numerical_results,
            "geometry": depth_geometry_results,
            "total_frames": len(depths)
        }
        
        # Count errors and warnings
        total_errors = sum(len(r[1]["errors"]) for r in depth_numerical_results + depth_geometry_results)
        total_warnings = sum(len(r[1]["warnings"]) for r in depth_numerical_results + depth_geometry_results)
        
        print(f"  Depth validation complete: {len(depths)} frame(s), {total_errors} error(s), {total_warnings} warning(s)")
    
    # Joint validation
    if not pose_only and not depth_only and len(poses) > 0 and len(depths) > 0:
        print("\nValidating geometric consistency...")
        
        # Get image size from the first depth map.
        if len(depths) > 0:
            height, width = depths[0].shape[:2]
        else:
            width, height = 1920, 1080  # default value
        
        geometry_result = validate_geometry_consistency(poses, focals, depths, width, height)
        results["geometry_validation"] = geometry_result
        
        total_errors = len(geometry_result.get("errors", []))
        total_warnings = len(geometry_result.get("warnings", []))
        print(f"  Geometric consistency validation complete: {total_errors} error(s), {total_warnings} warning(s)")
    
    # Build summary
    total_pose_errors = sum(len(r[1]["errors"]) for r in results.get("pose_validation", {}).get("format", [])) + \
                       sum(len(r[1]["errors"]) for r in results.get("pose_validation", {}).get("geometry", []))
    total_pose_warnings = sum(len(r[1]["warnings"]) for r in results.get("pose_validation", {}).get("format", [])) + \
                         sum(len(r[1]["warnings"]) for r in results.get("pose_validation", {}).get("geometry", []))
    
    total_depth_errors = sum(len(r[1]["errors"]) for r in results.get("depth_validation", {}).get("numerical", [])) + \
                        sum(len(r[1]["errors"]) for r in results.get("depth_validation", {}).get("geometry", []))
    total_depth_warnings = sum(len(r[1]["warnings"]) for r in results.get("depth_validation", {}).get("numerical", [])) + \
                          sum(len(r[1]["warnings"]) for r in results.get("depth_validation", {}).get("geometry", []))
    
    results["summary"] = {
        "total_frames": len(frame_numbers),
        "pose_errors": total_pose_errors,
        "pose_warnings": total_pose_warnings,
        "depth_errors": total_depth_errors,
        "depth_warnings": total_depth_warnings,
        "geometry_errors": len(results.get("geometry_validation", {}).get("errors", [])),
        "geometry_warnings": len(results.get("geometry_validation", {}).get("warnings", []))
    }
    
    # Visualization
    if visualize:
        print("\nGenerating visualizations...")
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = data_dir / "validation_vis"
            output_dir.mkdir(exist_ok=True)
        
        if not pose_only and len(poses) > 0:
            visualize_pose_trajectory(poses, output_dir / "pose_trajectory.png")
        
        if not depth_only and len(depths) > 0:
            visualize_depth_stats(depths, output_dir / "depth_stats.png")
    
    # Point cloud generation
    if export_pointcloud and not pose_only and not depth_only:
        if len(poses) > 0 and len(focals) > 0 and len(depths) > 0:
            print("\nPoint cloud generation...")
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir = data_dir / "validation_vis"
                output_dir.mkdir(exist_ok=True)
            
            # Try loading RGB images
            rgb_images = None
            if rgb_dir.exists() and HAS_PIL:
                try:
                    rgb_images = []
                    for frame_num in frame_numbers[:len(poses)]:
                        rgb_path = rgb_dir / f"{frame_num:06d}.png"
                        if rgb_path.exists():
                            img = Image.open(rgb_path)
                            rgb_images.append(np.array(img))
                        else:
                            rgb_images.append(None)
                    print(f"  Loaded {sum(1 for img in rgb_images if img is not None)} RGB image(s)")
                except Exception as e:
                    print(f"  Warning: failed to load RGB images: {e}")
                    rgb_images = None
            
            # Get image size.
            if len(depths) > 0:
                height, width = depths[0].shape[:2]
            else:
                width, height = 1920, 1080
            
            # Point cloud generation
            try:
                points, colors = generate_pointcloud(
                    poses, focals, depths,
                    rgb_images=rgb_images,
                    width=width, height=height,
                    downsample=pointcloud_downsample,
                    max_points=pointcloud_max_points,
                    depth_min=pointcloud_depth_min,
                    depth_max=pointcloud_depth_max
                )
                
                if len(points) > 0:
                    # Print point cloud stats.
                    print(f"  Point cloud stats: {len(points):,} points")
                    print(f"  Coordinate range:")
                    print(f"    X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
                    print(f"    Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
                    print(f"    Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
                    print(f"  Center: ({points[:, 0].mean():.3f}, {points[:, 1].mean():.3f}, {points[:, 2].mean():.3f})")
                    if colors is not None:
                        print(f"  Contains color information")
                    
                    # Save point cloud
                    ply_path = output_dir / "pointcloud.ply"
                    save_pointcloud_ply(points, ply_path, colors)
                else:
                    print("  Warning: cannot generate point cloud (no valid depth data)")
            except Exception as e:
                print(f"  Error: point cloud generation failed: {e}")
                import traceback
                traceback.print_exc()
    
    return results


def print_summary(results: Dict):
    """Print validation summary"""
    summary = results.get("summary", {})
    
    print("\n" + "=" * 60)
    print("Validation summary")
    print("=" * 60)
    print(f"Total frames: {summary.get('total_frames', 0)}")
    print(f"\nPose validation:")
    print(f"  Errors: {summary.get('pose_errors', 0)}")
    print(f"  Warnings: {summary.get('pose_warnings', 0)}")
    print(f"\nDepth validation:")
    print(f"  Errors: {summary.get('depth_errors', 0)}")
    print(f"  Warnings: {summary.get('depth_warnings', 0)}")
    print(f"\nGeometric consistency validation:")
    print(f"  Errors: {summary.get('geometry_errors', 0)}")
    print(f"  Warnings: {summary.get('geometry_warnings', 0)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Validate pose and depth data quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full validation
  python validate_pose_depth.py /path/to/data
  
  # Validate poses only
  python validate_pose_depth.py /path/to/data --pose-only
  
  # Validate depth only
  python validate_pose_depth.py /path/to/data --depth-only
  
  # Generate visualizations
  python validate_pose_depth.py /path/to/data --visualize
  
  # Save report
  python validate_pose_depth.py /path/to/data --output-report report.json
        """
    )
    
    parser.add_argument("data_dir", help="Path to data directory")
    parser.add_argument("--pose-only", action="store_true", help="Validate poses only")
    parser.add_argument("--depth-only", action="store_true", help="Validate depth only")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization images")
    parser.add_argument("--export-pointcloud", action="store_true", help="Export point cloud (PLY)")
    parser.add_argument("--pointcloud-downsample", type=int, default=2, help="Point cloud downsample factor (default: 2)")
    parser.add_argument("--pointcloud-max-points", type=int, default=None, help="Maximum point count for point cloud (default: unlimited)")
    parser.add_argument("--pointcloud-depth-min", type=float, default=None, help="Minimum depth value for filtering near points (camera Z)")
    parser.add_argument("--pointcloud-depth-max", type=float, default=None, help="Maximum depth value for filtering far points (camera Z)")
    parser.add_argument("--output-report", help="Save validation report to JSON file")
    parser.add_argument("--output-dir", help="Visualization output directory (default: data_dir/validation_vis)")
    
    args = parser.parse_args()
    
    try:
        results = validate_all(
            args.data_dir,
            pose_only=args.pose_only,
            depth_only=args.depth_only,
            visualize=args.visualize,
            output_dir=args.output_dir,
            export_pointcloud=args.export_pointcloud,
            pointcloud_downsample=args.pointcloud_downsample,
            pointcloud_max_points=args.pointcloud_max_points,
            pointcloud_depth_min=args.pointcloud_depth_min,
            pointcloud_depth_max=args.pointcloud_depth_max
        )
        
        print_summary(results)
        
        if args.output_report:
            with open(args.output_report, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nValidation report saved: {args.output_report}")
        
    except Exception as e:
        print(f"Errors: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
