# FoundationGeo Blender Data Pipeline

This repository provides a practical Blender-based pipeline for:
- Rendering RGB + depth from `.blend` scenes
- Exporting camera intrinsics/extrinsics (`focal` / `pose`)
- Converting depth EXR files into `.npy` and visualization PNGs

## Environment Setup

```bash
conda create -n py312 python=3.12
conda activate py312
pip install numpy matplotlib pillow OpenEXR tqdm
```

## Blender Setup

Recommended: **Blender 4.2 LTS**.

```bash
cd ~
wget https://download.blender.org/release/Blender4.2/blender-4.2.17-linux-x64.tar.xz
tar xf blender-4.2.17-linux-x64.tar.xz
```

Driver compatibility (NVIDIA):

| Blender Version | Minimum Driver |
|---|---|
| 5.0 | 535+ |
| 4.2 LTS | 470+ |
| 3.6 LTS | 450+ |

Check current driver:

```bash
nvidia-smi
```

## Quick Start

Unified entrypoint (`main.py`):

```bash
python main.py render scene/input.blend -o scene/ --export-animation
python main.py exr2all scene/depth/exr --colormap turbo
python main.py exr2npy scene/depth/exr --batch
python main.py exr2png scene/depth/exr --batch -c turbo
```

Standalone scripts are still available:

```bash
python scripts/render_and_convert.py scene/input.blend -o scene/ --export-animation
python scripts/depth_convert.py exr2npy scene/depth/exr --batch
python scripts/depth_convert.py exr2png scene/depth/exr --batch -c turbo
```

## YAML Config for Render

```bash
# Run render with config file (CLI flags override config)
python main.py render --config config.yaml scene/input.blend -o scene/

# Or set device directly on CLI
python main.py render scene/input.blend -o scene/ --device GPU --compute-type CUDA
```

Example config: `config.yaml`

Supported fields:
- `device`: `CPU` / `GPU`
- `compute_type`: `CUDA` / `OPTIX` / `HIP` / `METAL` / `ONEAPI`
- `gpu_ids`: `all` or comma list like `"0,1,2,3"`

## Auto Compositor Mode (`--no-compositor`)

By default, render scripts expect compositor nodes already configured in the `.blend` file.

Use `--no-compositor` to auto-create required nodes at runtime:

```bash
python main.py render input.blend -o scene/ \
  --export-animation \
  --device GPU \
  --compute-type CUDA \
  --no-compositor
```

This mode is useful when:
- Scenes do not have compositor setup
- You want quick rendering without manual node editing
- You are processing mixed `.blend` sources in batch

Auto-created nodes:
- Render Layers
- RGB File Output (PNG)
- Depth File Output (EXR 32-bit)

## Multi-GPU Rendering

### Option A: Single process, multiple GPUs

Best when each frame is expensive and you want all GPUs to render the same frame faster.

```bash
python main.py render input.blend -o scene/ \
  --device GPU \
  --compute-type CUDA \
  --gpu-ids all \
  --export-animation
```

### Option B: Multi-process parallel rendering

Best for animation sequences where each GPU renders different frame ranges.

```bash
python main.py parallel input.blend -o scene/ \
  --frame-start 1 --frame-end 240 \
  --num-gpus 8 \
  --compute-type CUDA
```

With 240 frames and 8 GPUs, frames are split automatically into contiguous ranges.

## Script Overview

- `main.py`: Unified entrypoint for render/convert/parallel workflows
- `scripts/cli.py`: Shared CLI argument definitions
- `scripts/render_and_convert.py`: CLI entry for render pipeline
- `scripts/render.py`: Core Blender-side rendering logic
- `scripts/export_camera.py`: Camera parameter export (`focal` / `pose`)
- `scripts/pipeline.py`: External Blender orchestration + live EXR conversion
- `scripts/parallel_render.py`: Multi-process multi-GPU rendering
- `scripts/config.py`: YAML config loading and merge
- `scripts/depth_convert.py`: EXR -> NPY/PNG (single or batch)

## Camera Export

Camera export writes:
- `focal/*.txt`: focal length(s) in pixels (`fx` or `fx fy`)
- `pose/*.txt`: 4x4 camera transform matrix

Single file:

```bash
blender --background --python scripts/export_camera.py -- \
  /path/to/input.blend \
  -o /path/to/output/
```

Batch mode:

```bash
python3 scripts/export_camera.py batch /path/to/blend_dir
```

Animation export produces frame-indexed files such as:
- `focal_000001.txt`
- `pose_000001.txt`

## Notes

1. Blender must be installed and available in `PATH`, or pass `--blender`.
2. The pipeline handles Blender-to-computer-vision coordinate conversion internally.
3. If no active camera exists, pass `-c/--camera` explicitly.
4. Batch mode skips backup files like `.blend1`.
