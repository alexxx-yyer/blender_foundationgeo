#!/usr/bin/env python3
"""读取 16bit PNG 深度图并解码为真实深度值（米）

用法:
  # 检查单个深度图
  python read_depth.py /path/to/depth.png

  # 检查 preprocessed 目录（验证多个样本）
  python read_depth.py --check-dir /path/to/processed/scene_name

  # 检查整个 processed 目录下的所有场景（多进程，只输出问题帧）
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
    读取 16bit PNG 深度图，解码为真实深度值（米）。
    
    编码方式（来自 preprocessed.py）:
        enc = 1 + round((log(depth/near) / log(far/near)) * 65533)
        特殊值: 0 = NaN, 65535 = inf
    
    解码公式:
        depth = near * (far/near)^((enc - 1) / 65533)
    
    返回:
        depth_meters: 真实深度数组（米），inf 表示无穷远
        near: 最小深度值
        far: 最大深度值
    """
    img = Image.open(depth_path)
    
    # 从 PNG 元数据读取 near 和 far
    near = float(img.text.get("near", "1e-5"))
    far = float(img.text.get("far", "1e4"))
    
    # 读取编码后的深度值
    enc = np.array(img, dtype=np.float32)
    
    # 解码为真实深度
    # depth = near * (far/near)^((enc - 1) / 65533)
    ratio = (enc - 1) / 65533.0
    depth_meters = near * np.power(far / near, ratio)
    
    # 处理特殊值
    depth_meters[enc == 0] = np.nan      # NaN
    depth_meters[enc == 65535] = np.inf  # 无穷远
    
    return depth_meters, near, far


def check_single_sample(sample_dir: str, verbose: bool = True, depth_threshold: float = 500) -> dict:
    """检查单个 preprocessed 样本的完整性和有效性。"""
    sample_dir = os.path.expanduser(sample_dir)
    result = {
        "path": sample_dir,
        "valid": True,
        "has_large_depth": False,
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    
    # 检查必需文件
    image_path = os.path.join(sample_dir, "image.jpg")
    depth_path = os.path.join(sample_dir, "depth.png")
    meta_path = os.path.join(sample_dir, "meta.json")
    
    for fpath, fname in [(image_path, "image.jpg"), (depth_path, "depth.png"), (meta_path, "meta.json")]:
        if not os.path.exists(fpath):
            result["valid"] = False
            result["errors"].append(f"缺少文件: {fname}")
    
    if not result["valid"]:
        return result
    
    # 检查深度图
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
        
        # 检查是否有超大深度值
        if result["stats"]["depth_max"] is not None and result["stats"]["depth_max"] > depth_threshold:
            result["has_large_depth"] = True
            result["warnings"].append(f"深度最大值过大: {result['stats']['depth_max']:.2f}m > {depth_threshold}m")
        
        if valid_ratio < 0.5:
            result["warnings"].append(f"有效深度比例过低: {valid_ratio*100:.1f}%")
        
        if near <= 0:
            result["errors"].append(f"near 值无效: {near}")
            result["valid"] = False
        
        if far <= near:
            result["errors"].append(f"far <= near: far={far}, near={near}")
            result["valid"] = False
            
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"深度图读取失败: {e}")
    
    # 检查 meta.json
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        if "intrinsics" not in meta:
            result["errors"].append("meta.json 缺少 intrinsics")
            result["valid"] = False
        if "camera_pose" not in meta:
            result["errors"].append("meta.json 缺少 camera_pose")
            result["valid"] = False
        
        result["stats"]["intrinsics"] = meta.get("intrinsics")
        result["stats"]["camera_pose"] = meta.get("camera_pose")
        
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"meta.json 读取失败: {e}")
    
    # 检查图像
    try:
        from PIL import Image as PILImage
        img = PILImage.open(image_path)
        result["stats"]["image_size"] = img.size
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"图像读取失败: {e}")
    
    if verbose and result["valid"]:
        stats = result["stats"]
        print(f"  ✓ {os.path.basename(sample_dir)}: "
              f"depth=[{stats['depth_min']:.2f}, {stats['depth_max']:.2f}]m, "
              f"valid={stats['valid_ratio']*100:.1f}%")
    elif verbose:
        print(f"  ✗ {os.path.basename(sample_dir)}: {', '.join(result['errors'])}")
    
    return result


def _check_sample_worker(args: tuple) -> dict:
    """多进程 worker：检查单个样本"""
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
    """检查一个场景目录下的所有样本。"""
    scene_dir = os.path.expanduser(scene_dir)
    scene_name = os.path.basename(scene_dir)
    
    # 查找所有样本目录
    sample_dirs = sorted(glob.glob(os.path.join(scene_dir, "*")))
    sample_dirs = [d for d in sample_dirs if os.path.isdir(d)]
    
    if max_samples and max_samples > 0:
        sample_dirs = sample_dirs[:max_samples]
    
    print(f"\n{'='*60}")
    print(f"场景: {scene_name} ({len(sample_dirs)} 个样本)")
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
    
    # 统计信息
    print(f"\n总计: {results['valid']}/{results['total']} 有效")
    if all_depths:
        print(f"深度范围: [{min(all_depths):.2f}, {max(all_depths):.2f}] 米")
    
    if results["invalid"] > 0:
        print(f"⚠ 无效样本: {results['invalid']} 个")
    if results["large_depth"] > 0:
        print(f"⚠ 超大深度样本: {results['large_depth']} 个")
    
    return results


def check_all_processed(processed_dir: str, max_samples_per_scene: int = 5, verbose: bool = True, 
                        depth_threshold: float = 500, num_workers: int = None) -> dict:
    """检查整个 processed 目录下的所有场景。"""
    processed_dir = os.path.expanduser(processed_dir)
    
    scene_dirs = sorted(glob.glob(os.path.join(processed_dir, "*")))
    scene_dirs = [d for d in scene_dirs if os.path.isdir(d) and not os.path.basename(d).startswith(".")]
    scene_names = [os.path.basename(d) for d in scene_dirs]
    
    # 收集所有样本目录，并记录每个样本属于哪个场景
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
    print(f"检查 processed 目录: {processed_dir}")
    print(f"场景数量: {len(scene_dirs)}")
    print(f"总样本数: {len(all_sample_dirs)}")
    print(f"深度阈值: {depth_threshold}m")
    print(f"进程数: {num_workers}")
    print(f"{'#'*60}\n")
    
    # 多进程检查
    tasks = [(d, depth_threshold) for d in all_sample_dirs]
    
    total_valid = 0
    total_invalid = 0
    total_large_depth = 0
    problem_samples = []  # 有问题的样本
    all_results = []  # 所有结果，用于按场景汇总
    
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
                    # 输出有问题的帧
                    rel_path = os.path.relpath(result["path"], processed_dir)
                    stats = result["stats"]
                    print(f"⚠ {rel_path}: depth=[{stats['depth_min']:.2f}, {stats['depth_max']:.2f}]m")
                    problem_samples.append(result)
            else:
                total_invalid += 1
                rel_path = os.path.relpath(result["path"], processed_dir)
                print(f"✗ {rel_path}: {', '.join(result['errors'])}")
                problem_samples.append(result)
            
            # 输出进度（每 1000 个或最后一个）
            if completed % 1000 == 0 or completed == total:
                print(f"进度: {completed}/{total} ({100*completed/total:.1f}%) | 问题帧: {len(problem_samples)}")
    
    # 按场景汇总统计
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
    
    # 输出每个场景的汇总
    print(f"\n{'#'*60}")
    print(f"各场景深度范围")
    print(f"{'#'*60}")
    print(f"{'场景':<45} {'样本数':<10} {'深度范围 (米)':<25} {'问题帧':<10}")
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
    print(f"总结")
    print(f"{'#'*60}")
    print(f"场景数: {len(scene_dirs)}")
    print(f"检查样本数: {total_valid + total_invalid}")
    print(f"有效: {total_valid}, 无效: {total_invalid}")
    print(f"超大深度 (>{depth_threshold}m): {total_large_depth}")
    
    if problem_samples:
        print(f"\n问题样本列表 ({len(problem_samples)} 个):")
        for res in problem_samples[:50]:  # 最多显示50个
            rel_path = os.path.relpath(res["path"], processed_dir)
            if res["valid"] and res.get("has_large_depth"):
                print(f"  ⚠ {rel_path}: max_depth={res['stats']['depth_max']:.2f}m")
            else:
                print(f"  ✗ {rel_path}: {', '.join(res['errors'])}")
        if len(problem_samples) > 50:
            print(f"  ... 还有 {len(problem_samples) - 50} 个问题样本")
    
    return {
        "processed_dir": processed_dir,
        "total_valid": total_valid,
        "total_invalid": total_invalid,
        "total_large_depth": total_large_depth,
        "problem_samples": problem_samples,
        "scene_stats": scene_stats
    }


def show_single_depth(depth_path: str, save_npy: bool = False):
    """显示单个深度图的详细信息。"""
    depth_meters, near, far = read_depth_png(depth_path)
    
    print("=" * 60)
    print("深度图信息:")
    print("=" * 60)
    print(f"文件路径: {depth_path}")
    print(f"图像尺寸: {depth_meters.shape[1]} x {depth_meters.shape[0]} (宽 x 高)")
    print(f"近平面 (near): {near:.6f} 米")
    print(f"远平面 (far): {far:.6f} 米")
    
    valid_mask = np.isfinite(depth_meters)
    valid_depths = depth_meters[valid_mask]
    
    print(f"\n有效像素数: {np.sum(valid_mask)} / {depth_meters.size} ({100*np.sum(valid_mask)/depth_meters.size:.2f}%)")
    print(f"NaN 像素数: {np.sum(np.isnan(depth_meters))}")
    print(f"Inf 像素数: {np.sum(np.isinf(depth_meters))}")
    
    if len(valid_depths) > 0:
        print(f"\n真实深度范围: [{valid_depths.min():.4f}, {valid_depths.max():.4f}] 米")
        print(f"真实深度均值: {valid_depths.mean():.4f} 米")
        print(f"真实深度中位数: {np.median(valid_depths):.4f} 米")
    
    # 示例像素
    print("\n" + "=" * 60)
    print("示例像素的真实深度值:")
    print("=" * 60)
    h, w = depth_meters.shape
    positions = [
        ("左上角", 10, 10),
        ("右上角", 10, w-10),
        ("中心", h//2, w//2),
        ("左下角", h-10, 10),
        ("右下角", h-10, w-10),
    ]
    
    for name, y, x in positions:
        val = depth_meters[y, x]
        if np.isnan(val):
            print(f"{name} ({x}, {y}): NaN (无效)")
        elif np.isinf(val):
            print(f"{name} ({x}, {y}): inf (无穷远)")
        else:
            print(f"{name} ({x}, {y}): {val:.4f} 米")
    
    # 读取 meta.json
    meta_path = os.path.join(os.path.dirname(depth_path), "meta.json")
    if os.path.exists(meta_path):
        print("\n" + "=" * 60)
        print("相机参数 (来自 meta.json):")
        print("=" * 60)
        with open(meta_path, "r") as f:
            meta = json.load(f)
            print(json.dumps(meta, indent=2))
    
    if save_npy:
        output_path = depth_path.replace(".png", "_meters.npy")
        np.save(output_path, depth_meters)
        print(f"\n真实深度数组已保存到: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="读取深度图并解码为真实深度值 / 检查 preprocessed 结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查单个深度图
  python read_depth.py /path/to/depth.png

  # 检查单个样本目录
  python read_depth.py --sample /path/to/processed/scene/000001

  # 检查一个场景目录
  python read_depth.py --scene /path/to/processed/scene_name

  # 检查整个 processed 目录（多进程，只输出问题帧）
  python read_depth.py --check-all /path/to/processed --max-samples 0 --workers 8

  # 检查所有帧，深度阈值 500m
  python read_depth.py --check-all /path/to/processed --max-samples 0 --depth-threshold 500
        """
    )
    parser.add_argument("depth_path", nargs="?", help="深度图路径")
    parser.add_argument("--sample", type=str, help="检查单个样本目录")
    parser.add_argument("--scene", type=str, help="检查一个场景目录")
    parser.add_argument("--check-all", type=str, dest="check_all", help="检查整个 processed 目录（多进程）")
    parser.add_argument("--max-samples", type=int, default=5, help="每场景最多检查的样本数 (默认: 5, 0=全部)")
    parser.add_argument("--workers", "-j", type=int, default=None, help="并行进程数 (默认: CPU数-1)")
    parser.add_argument("--depth-threshold", type=float, default=500, help="深度阈值（米），超过此值视为问题帧 (默认: 500)")
    parser.add_argument("--save-npy", action="store_true", help="保存解码后的深度到 .npy 文件")
    parser.add_argument("-q", "--quiet", action="store_true", help="静默模式，只显示错误")
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
        # 默认检查示例
        default_path = "/home/muxin/zhiyue/synthetic_data_focal_length/data/processed/blender-4.1-splash/000001/depth.png"
        if os.path.exists(default_path):
            show_single_depth(default_path, save_npy=args.save_npy)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
