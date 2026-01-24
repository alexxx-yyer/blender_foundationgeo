# 环境安装
```bash
conda create -n py312 python=3.12
conda activate py312
pip install numpy matplotlib pillow OpenEXR
```

# 使用方式
```bash
# 统一入口（推荐）
python main.py render scene/input.blend -o scene/ --export-animation
python main.py exr2all scene/depth/exr --colormap turbo
python main.py exr2npy scene/depth/exr --batch
python main.py exr2png scene/depth/exr --batch -c turbo

# 保留原有独立脚本
python scripts/render_and_convert.py scene/input.blend -o scene/ --export-animation
python scripts/exr2npy.py scene/depth/exr --batch
python scripts/exr2png.py scene/depth/exr --batch
```
