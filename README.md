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
python scripts/depth_convert.py exr2npy scene/depth/exr --batch
python scripts/depth_convert.py exr2png scene/depth/exr --batch -c turbo
```

# YAML 配置（render）
```bash
# 使用配置文件运行渲染（CLI 参数会覆盖配置）
python main.py render --config config.yaml

# CLI 直接指定渲染设备
python main.py render scene/input.blend -o scene/ --device GPU --compute-type CUDA
```

示例配置见 `config.yaml`。

配置新增字段：
- `device`: CPU / GPU
- `compute_type`: CUDA / OPTIX / HIP / METAL / ONEAPI

# 相机参数导出

## 功能
从 Blender .blend 文件中提取相机参数并导出为 `focal.txt` 和 `pose.txt` 文件。

## 脚本说明
- `scripts/export_camera.py`：单个 .blend 文件的相机参数导出脚本
- `scripts/batch_export_cameras.py`：批量处理目录中所有 .blend 文件的脚本

## 使用方法

### 单个文件处理
```bash
# 基本用法（需指定输出目录）
blender --background --python scripts/export_camera.py -- \
  /home/alex/projects/FoundationGeo/data/blender/blender-4.1-splash.blend \
  -o /home/alex/projects/FoundationGeo/data/blender/output/

# 指定相机名称
blender --background --python scripts/export_camera.py -- \
  /home/alex/projects/FoundationGeo/data/blender/blender-4.1-splash.blend \
  -c Camera

# 指定渲染尺寸
blender --background --python scripts/export_camera.py -- \
  /home/alex/projects/FoundationGeo/data/blender/blender-4.1-splash.blend \
  -w 1920 --height 1080

# 导出动画中每一帧的相机参数
blender --background --python scripts/export_camera.py -- \
  /home/alex/projects/FoundationGeo/data/blender/blender-4.1-splash.blend \
  --export-animation

# 导出指定帧范围的动画（每5帧导出一次）
blender --background --python scripts/export_camera.py -- \
  /home/alex/projects/FoundationGeo/data/blender/blender-4.1-splash.blend \
  --export-animation --frame-start 1 --frame-end 100 --frame-step 5
```

### 批量处理
```bash
# 处理目录中所有 .blend 文件（输出到各自文件同目录）
python3 scripts/batch_export_cameras.py /home/alex/projects/FoundationGeo/data/blender

# 指定统一输出目录
python3 scripts/batch_export_cameras.py /home/alex/projects/FoundationGeo/data/blender \
  -o /home/alex/projects/FoundationGeo/data/blender/camera_params/

# 指定相机名称和渲染尺寸
python3 scripts/batch_export_cameras.py /home/alex/projects/FoundationGeo/data/blender \
  -c Camera -w 1920 --height 1080

# 批量导出动画中每一帧的相机参数
python3 scripts/batch_export_cameras.py /home/alex/projects/FoundationGeo/data/blender \
  --export-animation
```

## 输出文件格式

### 单帧导出

#### focal.txt
焦距值（像素单位）：
- 如果 fx = fy：单行一个值
- 如果 fx ≠ fy：一行两个值（fx fy）

示例：
```
1234.567890
```
或
```
1234.567890 1234.567890
```

#### pose.txt
4x4 变换矩阵（从世界坐标到相机坐标），每行一个值，共 16 个值。

示例：
```
1.00000000 0.00000000 0.00000000 0.00000000
0.00000000 1.00000000 0.00000000 0.00000000
0.00000000 0.00000000 1.00000000 0.00000000
0.00000000 0.00000000 0.00000000 1.00000000
```

### 动画导出
当使用 `--export-animation` 参数时，每一帧会生成单独的文件：
- `focal_000001.txt`, `focal_000002.txt`, ...（每帧的焦距）
- `pose_000001.txt`, `pose_000002.txt`, ...（每帧的位姿矩阵）

文件名格式为 `focal_XXXXXX.txt` 和 `pose_XXXXXX.txt`，其中 `XXXXXX` 是6位数字的帧号（例如：000001, 000002）。

## 注意事项
1. 需要安装 Blender 并在 PATH 中可用
2. 脚本会自动处理 Blender 坐标系到计算机视觉坐标系的转换
3. 如果场景中没有活动相机，需要指定相机名称（`-c` 参数）
4. 批量处理脚本会自动跳过 `.blend1` 等备份文件
5. 动画导出时，脚本会自动检测场景的帧范围，也可以使用 `--frame-start` 和 `--frame-end` 指定范围
6. 使用 `--frame-step` 可以控制导出帧的间隔（例如：`--frame-step 5` 表示每5帧导出一次）
