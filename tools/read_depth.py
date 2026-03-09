#!/usr/bin/env python3
"""Read 16-bit PNG depth and decode to metric depth values

Usage:
  # Inspect a single depth map
  python read_depth.py /path/to/depth.png

  # Check a preprocessed directory (multiple samples)
  python read_depth.py --check-dir /path/to/processed/scene_name

  # Check all scenes under processed (multiprocess, report problematic frames only)
  python read_depth.py --check-all /path/to/processed --max-samples 0 --workers 8
"""

import os
import glob
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count


def read_depth_png(depth_path: str) -> tuple[np.ndarray, float, float]:
    """
    Read a 16-bit PNG depth map and decode to metric depth values.
    
    Encoding scheme (from preprocessed.py):
        enc = 1 + round((log(depth/near) / log(far/near)) * 65533)
        Special values: 0 = NaN, 65535 = inf
    
    Decoding formula:
        depth = near * (far/near)^((enc - 1) / 65533)
    
    Returns:
        depth_meters: Metric depth array (meters), with inf for infinity
        near: minimum depth value
        far: maximum depth value
    """
    img = Image.open(depth_path)
    
    # Read near/far from PNG metadata
    near = float(img.text.get("near", "1e-5"))
    far = float(img.text.get("far", "1e4"))
    
    # Read encoded depth values
    enc = np.array(img, dtype=np.float32)
    
    # Decode to metric depth
    # depth = near * (far/near)^((enc - 1) / 65533)
    ratio = (enc - 1) / 65533.0
    depth_meters = near * np.power(far / near, ratio)
    
    # Handle special values
    depth_meters[enc == 0] = np.nan      # NaN
    depth_meters[enc == 65535] = np.inf  # infinity
    
    return depth_meters, near, far


def check_single_sample(sample_dir: str, verbose: bool = True, depth_threshold: float = 500) -> dict:
    """Check integrity and validity for one preprocessed sample."""
    sample_dir = os.path.expanduser(sample_dir)
    result = {
        "path": sample_dir,
        "valid": True,
        "has_large_depth": False,
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    
    # Check required files
    image_path = os.path.join(sample_dir, "image.jpg")
    depth_path = os.path.join(sample_dir, "depth.png")
    meta_path = os.path.join(sample_dir, "meta.json")
    
    for fpath, fname in [(image_path, "image.jpg"), (depth_path, "depth.png"), (meta_path, "meta.json")]:
        if not os.path.exists(fpath):
            result["valid"] = False
            result["errors"].append(f"Missing file: {fname}")
    
    if not result["valid"]:
        return result
    
    # Check depth map
    try:
        depth_meters, near, far = read_depth_png(depth_path)
        valid_mask = np.isfinite(depth_meters)
        valid_ratio = np.sum(valid_mask) / depth_meters.size
        
        result["stats"]["depth_shape"] = depth_meters.shape
        result["stats"]["near"] = near
        result["stats"]["far"] = far
        result["stats"]["valid_ratio"] = valid_ratio
        result["stats"]["depth_min"] = float(np.min(depth_meters[valid_mask])) if np.any(valid_mask) else None
        result["stats"]["depth_max"] = float(np.max(depth_meters[valid_mask])) if np.any(valid_mask) else None
        result["stats"]["depth_mean"] = float(np.mean(depth_meters[valid_mask])) if np.any(valid_mask) else None
        
        # Check whether max depth exceeds threshold.
        if result["stats"]["depth_max"] is not None and result["stats"]["depth_max"] > depth_threshold:
            result["has_large_depth"] = True
            result["warnings"].append(f"Depth max too large: {result['stats']['depth_max']:.2f}m > {depth_threshold}m")
        
        if valid_ratio < 0.5:
            result["warnings"].append(f"Low valid-depth ratio: {valid_ratio*100:.1f}%")
        
        if near <= 0:
            result["errors"].append(f"Invalid near value: {near}")
            result["valid"] = False
        
        if far <= near:
            result["errors"].append(f"far <= near: far={far}, near={near}")
            result["valid"] = False
            
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Failed to read depth map: {e}")
    
    # Check meta.json
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        if "intrinsics" not in meta:
            result["errors"].append("meta.json missing intrinsics")
            result["valid"] = False
        if "camera_pose" not in meta:
            result["errors"].append("meta.json missing camera_pose")
            result["valid"] = False
        
        result["stats"]["intrinsics"] = meta.get("intrinsics")
        result["stats"]["camera_pose"] = meta.get("camera_pose")
        
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Failed to read meta.json: {e}")
    
    # Check image
    try:
        from PIL import Image as PILImage
        img = PILImage.open(image_path)
        result["stats"]["image_size"] = img.size
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Failed to read image: {e}")
    
    if verbose and result["valid"]:
        stats = result["stats"]
        print(f"  ✓ {os.path.basename(sample_dir)}: "
              f"depth=[{stats['depth_min']:.2f}, {stats['depth_max']:.2f}]m, "
              f"valid={stats['valid_ratio']*100:.1f}%")
    elif verbose:
        print(f"  ✗ {os.path.basename(sample_dir)}: {', '.join(result['errors'])}")
    
    return result


def _check_sample_worker(args: tuple) -> dict:
    """Multiprocessing worker: check one sample"""
    sample_dir, depth_threshold = args
    try:
        return check_single_sample(sample_dir, verbose=False, depth_threshold=depth_threshold)
    except Exception as e:
        return {
            "path": sample_dir,
            "valid": False,
            "has_large_depth": False,
            "errors": [str(e)],
            "warnings": [],
            "stats": {}
        }


def check_scene_dir(scene_dir: str, max_samples: int = None, verbose: bool = True, depth_threshold: float = 500) -> dict:
    """Check all samples in one scene directory."""
    scene_dir = os.path.expanduser(scene_dir)
    scene_name = os.path.basename(scene_dir)
    
    # Find all sample directories
    sample_dirs = sorted(glob.glob(os.path.join(scene_dir, "*")))
    sample_dirs = [d for d in sample_dirs if os.path.isdir(d)]
    
    if max_samples and max_samples > 0:
        sample_dirs = sample_dirs[:max_samples]
    
    print(f"\n{'='*60}")
    print(f"Scene: {scene_name} ({len(sample_dirs)} sample(s))")
    print(f"{'='*60}")
    
    results = {
        "scene": scene_name,
        "total": len(sample_dirs),
        "valid": 0,
        "invalid": 0,
        "large_depth": 0,
        "samples": []
    }
    
    all_depths = []
    for sample_dir in sample_dirs:
        res = check_single_sample(sample_dir, verbose=verbose, depth_threshold=depth_threshold)
        results["samples"].append(res)
        if res["valid"]:
            results["valid"] += 1
            if res["stats"].get("depth_min") is not None:
                all_depths.extend([res["stats"]["depth_min"], res["stats"]["depth_max"]])
            if res.get("has_large_depth"):
                results["large_depth"] += 1
        else:
            results["invalid"] += 1
    
    # Statistics
    print(f"\nTotal: {results['valid']}/{results['total']} valid")
    if all_depths:
        print(f"Depth range: [{min(all_depths):.2f}, {max(all_depths):.2f}] m")
    
    if results["invalid"] > 0:
        print(f"⚠ Invalid samples: {results['invalid']}")
    if results["large_depth"] > 0:
        print(f"⚠ Large-depth samples: {results['large_depth']}")
    
    return results


def check_all_processed(processed_dir: str, max_samples_per_scene: int = 5, verbose: bool = True, 
                        depth_threshold: float = 500, num_workers: int = None) -> dict:
    """Check all scenes under the processed directory."""
    processed_dir = os.path.expanduser(processed_dir)
    
    scene_dirs = sorted(glob.glob(os.path.join(processed_dir, "*")))
    scene_dirs = [d for d in scene_dirs if os.path.isdir(d) and not os.path.basename(d).startswith(".")]
    scene_names = [os.path.basename(d) for d in scene_dirs]
    
    # Collect all sample directories and map each sample to a scene.
    all_sample_dirs = []
    sample_to_scene = {}  # sample_dir -> scene_name
    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        sample_dirs = sorted(glob.glob(os.path.join(scene_dir, "*")))
        sample_dirs = [d for d in sample_dirs if os.path.isdir(d)]
        if max_samples_per_scene and max_samples_per_scene > 0:
            sample_dirs = sample_dirs[:max_samples_per_scene]
        for sd in sample_dirs:
            sample_to_scene[sd] = scene_name
        all_sample_dirs.extend(sample_dirs)
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    print(f"\n{'#'*60}")
    print(f"Checking processed directory: {processed_dir}")
    print(f"Scene count: {len(scene_dirs)}")
    print(f"Total samples: {len(all_sample_dirs)}")
    print(f"Depth threshold: {depth_threshold}m")
    print(f"Workers: {num_workers}")
    print(f"{'#'*60}\n")
    
    # Multiprocess validation
    tasks = [(d, depth_threshold) for d in all_sample_dirs]
    
    total_valid = 0
    total_invalid = 0
    total_large_depth = 0
    problem_samples = []  # Problematic samples
    all_results = []  # All results for per-scene aggregation
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_check_sample_worker, task): task for task in tasks}
        
        completed = 0
        total = len(futures)
        
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            all_results.append(result)
            
            if result["valid"]:
                total_valid += 1
                if result.get("has_large_depth"):
                    total_large_depth += 1
                    # Report problematic frames
                    rel_path = os.path.relpath(result["path"], processed_dir)
                    stats = result["stats"]
                    print(f"⚠ {rel_path}: depth=[{stats['depth_min']:.2f}, {stats['depth_max']:.2f}]m")
                    problem_samples.append(result)
            else:
                total_invalid += 1
                rel_path = os.path.relpath(result["path"], processed_dir)
                print(f"✗ {rel_path}: {', '.join(result['errors'])}")
                problem_samples.append(result)
            
            # Print progress (every 1000 or final)
            if completed % 1000 == 0 or completed == total:
                print(f"Progress: {completed}/{total} ({100*completed/total:.1f}%) | Problem frames: {len(problem_samples)}")
    
    # Per-scene summary
    scene_stats = {name: {"valid": 0, "invalid": 0, "large_depth": 0, "depth_min": [], "depth_max": []} 
                   for name in scene_names}
    
    for result in all_results:
        scene_name = sample_to_scene.get(result["path"])
        if scene_name is None:
            continue
        
        if result["valid"]:
            scene_stats[scene_name]["valid"] += 1
            if result["stats"].get("depth_min") is not None:
                scene_stats[scene_name]["depth_min"].append(result["stats"]["depth_min"])
                scene_stats[scene_name]["depth_max"].append(result["stats"]["depth_max"])
            if result.get("has_large_depth"):
                scene_stats[scene_name]["large_depth"] += 1
        else:
            scene_stats[scene_name]["invalid"] += 1
    
    # Print summary for each scene
    print(f"\n{'#'*60}")
    print(f"Per-scene depth ranges")
    print(f"{'#'*60}")
    print(f"{'Scene':<45} {'Samples':<10} {'Depth range (m)':<25} {'Problem frames':<10}")
    print("-" * 90)
    
    for scene_name in scene_names:
        stats = scene_stats[scene_name]
        total_samples = stats["valid"] + stats["invalid"]
        if stats["depth_min"] and stats["depth_max"]:
            depth_range = f"[{min(stats['depth_min']):.2f}, {max(stats['depth_max']):.2f}]"
        else:
            depth_range = "N/A"
        
        problem_count = stats["large_depth"] + stats["invalid"]
        problem_str = str(problem_count) if problem_count > 0 else "-"
        
        print(f"{scene_name:<45} {total_samples:<10} {depth_range:<25} {problem_str:<10}")
    
    print(f"\n{'#'*60}")
    print(f"Summary")
    print(f"{'#'*60}")
    print(f"Scene count: {len(scene_dirs)}")
    print(f"Checked samples: {total_valid + total_invalid}")
    print(f"Valid: {total_valid}, invalid: {total_invalid}")
    print(f"Large depth (>{depth_threshold}m): {total_large_depth}")
    
    if problem_samples:
        print(f"\nProblem sample list ({len(problem_samples)}):")
        for res in problem_samples[:50]:  # Show up to 50 entries.
            rel_path = os.path.relpath(res["path"], processed_dir)
            if res["valid"] and res.get("has_large_depth"):
                print(f"  ⚠ {rel_path}: max_depth={res['stats']['depth_max']:.2f}m")
            else:
                print(f"  ✗ {rel_path}: {', '.join(res['errors'])}")
        if len(problem_samples) > 50:
            print(f"  ... and {len(problem_samples) - 50} more problem sample(s)")
    
    return {
        "processed_dir": processed_dir,
        "total_valid": total_valid,
        "total_invalid": total_invalid,
        "total_large_depth": total_large_depth,
        "problem_samples": problem_samples,
        "scene_stats": scene_stats
    }


def show_single_depth(depth_path: str, save_npy: bool = False):
    """Show detailed info for one depth map."""
    depth_meters, near, far = read_depth_png(depth_path)
    
    print("=" * 60)
    print("Depth map info:")
    print("=" * 60)
    print(f"File path: {depth_path}")
    print(f"Image size: {depth_meters.shape[1]} x {depth_meters.shape[0]} (W x H)")
    print(f"Near plane (near): {near:.6f} m")
    print(f"Far plane (far): {far:.6f} m")
    
    valid_mask = np.isfinite(depth_meters)
    valid_depths = depth_meters[valid_mask]
    
    print(f"\nValid pixels: {np.sum(valid_mask)} / {depth_meters.size} ({100*np.sum(valid_mask)/depth_meters.size:.2f}%)")
    print(f"NaN pixels: {np.sum(np.isnan(depth_meters))}")
    print(f"Inf pixels: {np.sum(np.isinf(depth_meters))}")
    
    if len(valid_depths) > 0:
        print(f"\nDepth range: [{valid_depths.min():.4f}, {valid_depths.max():.4f}] m")
        print(f"Depth mean: {valid_depths.mean():.4f} m")
        print(f"Depth median: {np.median(valid_depths):.4f} m")
    
    # Example pixels
    print("\n" + "=" * 60)
    print("Metric depth at example pixels:")
    print("=" * 60)
    h, w = depth_meters.shape
    positions = [
        ("Top-left", 10, 10),
        ("Top-right", 10, w-10),
        ("Center", h//2, w//2),
        ("Bottom-left", h-10, 10),
        ("Bottom-right", h-10, w-10),
    ]
    
    for name, y, x in positions:
        val = depth_meters[y, x]
        if np.isnan(val):
            print(f"{name} ({x}, {y}): NaN (invalid)")
        elif np.isinf(val):
            print(f"{name} ({x}, {y}): inf (infinity)")
        else:
            print(f"{name} ({x}, {y}): {val:.4f} m")
    
    # Read meta.json
    meta_path = os.path.join(os.path.dirname(depth_path), "meta.json")
    if os.path.exists(meta_path):
        print("\n" + "=" * 60)
        print("Camera parameters (from meta.json):")
        print("=" * 60)
        with open(meta_path, "r") as f:
            meta = json.load(f)
            print(json.dumps(meta, indent=2))
    
    if save_npy:
        output_path = depth_path.replace(".png", "_meters.npy")
        np.save(output_path, depth_meters)
        print(f"\nSaved decoded depth array to: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Read depth PNG, decode metric depth, and validate preprocessed outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect a single depth map
  python read_depth.py /path/to/depth.png

  # Check a single sample directory
  python read_depth.py --sample /path/to/processed/scene/000001

  # Check one scene directory
  python read_depth.py --scene /path/to/processed/scene_name

  # Check the full processed directory (multiprocess, problem frames only)
  python read_depth.py --check-all /path/to/processed --max-samples 0 --workers 8

  # Check all frames with depth threshold = 500m
  python read_depth.py --check-all /path/to/processed --max-samples 0 --depth-threshold 500
        """
    )
    parser.add_argument("depth_path", nargs="?", help="Path to depth image")
    parser.add_argument("--sample", type=str, help="Check a single sample directory")
    parser.add_argument("--scene", type=str, help="Check one scene directory")
    parser.add_argument("--check-all", type=str, dest="check_all", help="Check full processed directory (multiprocess)")
    parser.add_argument("--max-samples", type=int, default=5, help="Max samples per scene (default: 5, 0=all)")
    parser.add_argument("--workers", "-j", type=int, default=None, help="Worker count (default: CPU cores - 1)")
    parser.add_argument("--depth-threshold", type=float, default=500, help="Depth threshold in meters for flagging problem frames (default: 500)")
    parser.add_argument("--save-npy", action="store_true", help="Save decoded depth to .npy")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode (errors only)")
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    if args.check_all:
        check_all_processed(
            args.check_all, 
            max_samples_per_scene=args.max_samples, 
            verbose=verbose,
            depth_threshold=args.depth_threshold,
            num_workers=args.workers
        )
    elif args.scene:
        check_scene_dir(args.scene, max_samples=args.max_samples, verbose=verbose, depth_threshold=args.depth_threshold)
    elif args.sample:
        check_single_sample(args.sample, verbose=True, depth_threshold=args.depth_threshold)
    elif args.depth_path:
        show_single_depth(args.depth_path, save_npy=args.save_npy)
    else:
        # Default demo check
        default_path = "/home/muxin/zhiyue/synthetic_data_focal_length/data/processed/blender-4.1-splash/000001/depth.png"
        if os.path.exists(default_path):
            show_single_depth(default_path, save_npy=args.save_npy)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
