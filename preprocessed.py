#!/usr/bin/env python3
"""
 synthetic_data_focal_length/data/rendered  data/processed 。
: rendered/<scene>/rgb, focal, pose, depth(exr|npy)
: processed/<scene>/<frame_id>/image.jpg, depth.png, meta.json
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Tuple
import warnings
from PIL import Image, PngImagePlugin

warnings.filterwarnings("ignore")

# defaultpath： rendered， processed
RENDERED_DIR = "/home/muxin/zhiyue/synthetic_data_focal_length/data/rendered"
OUTPUT_DIR = "/home/muxin/zhiyue/synthetic_data_focal_length/data/processed"
NUM_WORKERS = max(1, (os.cpu_count() or 1) - 1)
FORCE_OVERWRITE = False  # ，force overwritefile


def write_image(path, image: np.ndarray, quality: int = 95):
    """write RGB  (H,W,3) uint8."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    enc = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
    cv2.imwrite(str(path), enc, [cv2.IMWRITE_JPEG_QUALITY, quality])


def write_meta(path, data: dict):
    """write meta.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=None, separators=(",", ":"))


def write_depth(path, depth: np.ndarray, max_range: float = 1e5):
    """depth  16bit PNG，log ，near/far  PNG （）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_values = np.isfinite(depth)
    mask_nan, mask_inf = np.isnan(depth), np.isinf(depth)
    depth = depth.astype(np.float32)
    if not np.any(mask_values):
        near, far = 1e-5, 1e4
    else:
        near = max(float(np.min(depth[mask_values])), 1e-5)
        far = max(near * 1.1, min(float(np.max(depth[mask_values])), near * max_range))
    enc = 1 + np.round((np.log(np.nan_to_num(depth, nan=0).clip(near, far) / near) / np.log(far / near)).clip(0, 1) * 65533).astype(np.uint16)
    enc[mask_nan], enc[mask_inf] = 0, 65535
    img = Image.fromarray(enc)
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("near", str(near))
    pnginfo.add_text("far", str(far))
    img.save(path, pnginfo=pnginfo, compress_level=7)


def load_focal(focal_path: Path) -> Tuple[float, float]:
    """focal/<frame>.txt:  fx [fy]（）； fx=fy。"""
    with open(focal_path) as f:
        line = f.read().strip().split()
    fx = float(line[0])
    fy = float(line[1]) if len(line) > 1 else fx
    return fx, fy


def load_pose(pose_path: Path) -> np.ndarray:
    """pose/<frame>.txt: 4 ， 4 ，c2w 4x4."""
    with open(pose_path) as f:
        rows = [list(map(float, line.strip().split())) for line in f if line.strip()]
    return np.array(rows, dtype=np.float32)


def load_depth_exr(exr_path: Path) -> np.ndarray:
    """ EXR  (float32)， inf。 OpenEXR。"""
    import OpenEXR  # type: ignore
    import Imath  # type: ignore
    exr = OpenEXR.InputFile(str(exr_path))
    header = exr.header()
    dw = header["dataWindow"]
    w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    chans = list(header["channels"].keys())
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    data = {}
    for c in chans:
        raw = exr.channel(c, pt)
        data[c] = np.frombuffer(raw, dtype=np.float32).reshape(h, w)
    if "Z" in data:
        depth = data["Z"].copy()
    elif "depth" in data:
        depth = data["depth"].copy()
    elif len(chans) == 1:
        depth = data[chans[0]].copy()
    else:
        depth = data[sorted(chans)[0]].copy()
    depth[depth >= 500] = np.inf
    return depth


def load_depth_npy(npy_path: Path) -> np.ndarray:
    """ NPY ； Z。"""
    d = np.load(npy_path).astype(np.float32)
    if d.ndim == 3:
        if d.shape[-1] >= 3 and "Z" in str(npy_path):
            d = d[:, :, 0]
        else:
            d = d.squeeze()
    if d.ndim != 2:
        d = d.squeeze()
    d[d >= 500] = np.inf  #  load_depth_exr ， inf
    return d


def depth_exists(scene_rendered: Path, frame_id: str) -> bool:
    """file，read。。"""
    depth_dir = scene_rendered / "depth"
    return (depth_dir / "exr" / f"{frame_id}.exr").exists() or (depth_dir / "npy" / f"{frame_id}.npy").exists()


def load_depth(scene_rendered: Path, frame_id: str) -> Optional[np.ndarray]:
    """ exr（ OpenEXR）， npy。"""
    depth_dir = scene_rendered / "depth"
    exr_f = depth_dir / "exr" / f"{frame_id}.exr"
    npy_f = depth_dir / "npy" / f"{frame_id}.npy"
    if exr_f.exists():
        try:
            return load_depth_exr(exr_f)
        except ImportError:
            pass  #  OpenEXR  npy
    if npy_f.exists():
        return load_depth_npy(npy_f)
    return None


def build_intrinsics(fx_px: float, fy_px: float, width: int, height: int) -> np.ndarray:
    """Resolution 3x3 K，。"""
    K = np.eye(3, dtype=np.float32)
    K[0, 0], K[1, 1] = fx_px, fy_px
    K[0, 2], K[1, 2] = width / 2.0, height / 2.0
    return K


def normalize_intrinsics(K: np.ndarray, width: int, height: int) -> np.ndarray:
    """：fx_norm=fx/W, cx_norm=cx/W 。"""
    Kn = np.eye(3, dtype=np.float32)
    Kn[0, 0], Kn[0, 2] = K[0, 0] / width, K[0, 2] / width
    Kn[1, 1], Kn[1, 2] = K[1, 1] / height, K[1, 2] / height
    return Kn


def write_index_file(output_dir: Path, instance_paths: List[str]):
    """write .index.txt，。 dataloader default .index.txt。"""
    existing = []
    for rel in instance_paths:
        d = output_dir / rel
        if (d / "image.jpg").exists() and (d / "depth.png").exists() and (d / "meta.json").exists():
            existing.append(rel)
    if existing:
        (output_dir / ".index.txt").write_text("\n".join(sorted(existing)) + "\n")
        print(f"   .index.txt， {len(existing)} ")


def process_frame(args: Tuple[Path, Path, str, Path]) -> Optional[str]:
    """：(scene_rendered, output_base, frame_id, rgb_path) -> instance_path or None."""
    scene_rendered, output_base, frame_id, rgb_path = args
    out_dir = output_base / frame_id
    # force overwrite，file，
    if not FORCE_OVERWRITE and (out_dir / "image.jpg").exists() and (out_dir / "depth.png").exists() and (out_dir / "meta.json").exists():
        return f"{output_base.name}/{frame_id}"

    rgb = cv2.imread(str(rgb_path))
    if rgb is None:
        return None
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    focal_path = scene_rendered / "focal" / f"{frame_id}.txt"
    pose_path = scene_rendered / "pose" / f"{frame_id}.txt"
    if not focal_path.exists() or not pose_path.exists():
        return None

    fx_px, fy_px = load_focal(focal_path)
    c2w = load_pose(pose_path)
    depth = load_depth(scene_rendered, frame_id)
    if depth is None:
        return None

    if depth.shape[0] != h or depth.shape[1] != w:
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    K = build_intrinsics(fx_px, fy_px, w, h)
    K_norm = normalize_intrinsics(K, w, h)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_image(out_dir / "image.jpg", rgb, quality=95)
    write_depth(out_dir / "depth.png", depth)
    write_meta(out_dir / "meta.json", {
        "intrinsics": K_norm.tolist(),
        "camera_pose": c2w.tolist(),
    })
    return f"{output_base.name}/{frame_id}"


def _run_one(t) -> Optional[str]:
    """ worker：， None。"""
    try:
        return process_frame(t)
    except Exception:
        return None


def main(rendered_dir: str = RENDERED_DIR, output_dir: str = OUTPUT_DIR, num_workers: int = NUM_WORKERS, force: bool = False):
    global FORCE_OVERWRITE
    FORCE_OVERWRITE = force
    
    rendered_dir = Path(rendered_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if force:
        print("⚠ force overwrite：", flush=True)

    # Scene
    scene_names = [
        d.name for d in rendered_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name != "mlruns"
    ]
    print(f" {len(scene_names)} Scene、...", flush=True)

    # 
    all_tasks: List[tuple] = []
    for scene_name in tqdm(scene_names, desc="Scene"):
        scene_rendered = rendered_dir / scene_name
        rgb_dir = scene_rendered / "rgb"
        if not rgb_dir.is_dir():
            continue
        for rgb_path in sorted(rgb_dir.glob("*.png")):
            frame_id = rgb_path.stem
            if not (scene_rendered / "focal" / f"{frame_id}.txt").exists():
                continue
            if not (scene_rendered / "pose" / f"{frame_id}.txt").exists():
                continue
            if not depth_exists(scene_rendered, frame_id):
                continue
            all_tasks.append((scene_rendered, output_dir / scene_name, frame_id, rgb_path))

    all_instance_paths: List[str] = []
    if not all_tasks:
        write_index_file(output_dir, all_instance_paths)
        print("Done: ")
        return

    n_total = len(all_tasks)
    print(f" {n_total} ，...", flush=True)
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futs = {ex.submit(_run_one, t): t for t in all_tasks}
        done = 0
        for fut in tqdm(as_completed(futs), total=n_total, desc="frames"):
            rel = fut.result()
            if rel:
                all_instance_paths.append(rel)
            done += 1
            if done % 500 == 0 or done == n_total:
                print(f"  Done {done}/{n_total}", flush=True)

    write_index_file(output_dir, all_instance_paths)
    print(f"Done: Output directory {output_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="rendered -> processed")
    p.add_argument("--input", "-i", default=RENDERED_DIR, help="rendered root directory")
    p.add_argument("--output", "-o", default=OUTPUT_DIR, help="processed Output directory")
    p.add_argument("--workers", "-j", type=int, default=NUM_WORKERS, help="worker count，default cpu  - 1")
    p.add_argument("--force", "-f", action="store_true", help="force overwritefile（）")
    args = p.parse_args()
    main(rendered_dir=args.input, output_dir=args.output, num_workers=args.workers, force=args.force)
