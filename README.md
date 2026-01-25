# 环境安装
```bash
conda create -n py312 python=3.12
conda activate py312
pip install numpy matplotlib pillow OpenEXR tqdm
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

配置字段：
- `device`: CPU / GPU
- `compute_type`: CUDA / OPTIX / HIP / METAL / ONEAPI
- `gpu_ids`: all / "0,1,2,3,4,5,6,7"（指定使用的 GPU）

# 多 GPU 渲染

## 方案一：单进程多卡（多张 GPU 同时渲染同一帧）

适合场景复杂、单帧渲染很慢的情况。Blender Cycles 会将渲染任务分配到所有启用的 GPU 上。

```bash
# 使用配置文件（gpu_ids: all 启用所有 GPU）
python main.py render --config config.yaml input.blend

# 命令行指定
python main.py render input.blend -o scene/ \
    --device GPU \
    --compute-type CUDA \
    --gpu-ids all \
    --export-animation

# 只使用前 4 张卡
python main.py render input.blend -o scene/ \
    --device GPU \
    --compute-type CUDA \
    --gpu-ids "0,1,2,3" \
    --export-animation
```

配置文件示例 (`config.yaml`)：
```yaml
device: GPU
compute_type: CUDA
gpu_ids: all  # 或 "0,1,2,3,4,5,6,7"
```

## 方案二：多进程并行（每张 GPU 渲染不同帧）

适合多帧动画渲染，理论上可达到单卡 N 倍的渲染速度（N = GPU 数量）。

```bash
# 8 张卡并行渲染 1-240 帧
python main.py parallel input.blend -o scene/ \
    --frame-start 1 --frame-end 240 \
    --num-gpus 8 \
    --compute-type CUDA

# 4 张卡并行，使用 OPTIX
python main.py parallel input.blend -o scene/ \
    --frame-start 1 --frame-end 100 \
    --num-gpus 4 \
    --compute-type OPTIX

# 完整参数示例
python main.py parallel input.blend -o scene/ \
    --frame-start 1 --frame-end 240 \
    --num-gpus 8 \
    --frame-step 1 \
    --compute-type CUDA \
    -c Camera \
    -w 1920 --height 1080 \
    --colormap turbo
```

帧自动分配示例（240 帧 / 8 卡）：
```
GPU 0: 帧 1-30   (30帧)
GPU 1: 帧 31-60  (30帧)
GPU 2: 帧 61-90  (30帧)
GPU 3: 帧 91-120 (30帧)
GPU 4: 帧 121-150 (30帧)
GPU 5: 帧 151-180 (30帧)
GPU 6: 帧 181-210 (30帧)
GPU 7: 帧 211-240 (30帧)
```

## 方案选择建议

| 场景 | 推荐方案 |
|------|---------|
| 单帧渲染慢（>1分钟/帧） | 方案一：单进程多卡 |
| 多帧动画渲染 | 方案二：多进程并行 |
| 帧数少但单帧复杂 | 方案一：单进程多卡 |
| 帧数多且单帧较快 | 方案二：多进程并行 |

# 脚本总览（Python）
- `main.py`：统一入口；解析子命令并调度渲染/转换；支持 YAML 配置与设备选择
- `scripts/cli.py`：CLI 参数构建（render/parallel 与 main 共用）
- `scripts/render_and_convert.py`：渲染 CLI 入口（调用 `scripts/pipeline.py`）
- `scripts/render.py`：Blender 内部渲染核心（RGB/Depth EXR、节点/帧循环/进度、多 GPU 设备设置）
- `scripts/export_camera.py`：导出 focal/pose（单帧/动画/批量子命令）
- `scripts/pipeline.py`：渲染管线（外部 Blender 调度 + EXR 实时转换）
- `scripts/parallel_render.py`：多进程并行渲染（每张 GPU 渲染不同帧范围）
- `scripts/config.py`：YAML 配置读取与合并
- `scripts/depth_convert.py`：EXR 转 NPY/PNG（含批量与 exr2all）
- `scripts/read_npy.py`：读取 NPY 并打印统计/可视化
- `scripts/__init__.py`：scripts 包标记

# 功能重复点（可优化）
- CLI 参数定义在 `main.py` 与 `scripts/cli.py` 已统一为 `scripts/cli.py`，避免重复（当前无重复）
- 渲染/相机导出分层：`render.py` + `export_camera.py` 被 `render_and_convert.py` 组合调用（属封装，不算逻辑重复）
- 批量相机导出已合并进 `scripts/export_camera.py batch`（无重复）

# 相机参数导出

## 功能
从 Blender .blend 文件中提取相机参数并导出为 `focal.txt` 和 `pose.txt` 文件。

## 脚本说明
- `scripts/export_camera.py`：单个/批量 .blend 文件的相机参数导出脚本

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
python3 scripts/export_camera.py batch /home/alex/projects/FoundationGeo/data/blender

# 指定统一输出目录
python3 scripts/export_camera.py batch /home/alex/projects/FoundationGeo/data/blender \
  -o /home/alex/projects/FoundationGeo/data/blender/camera_params/

# 指定相机名称和渲染尺寸
python3 scripts/export_camera.py batch /home/alex/projects/FoundationGeo/data/blender \
  -c Camera -w 1920 --height 1080

# 批量导出动画中每一帧的相机参数
python3 scripts/export_camera.py batch /home/alex/projects/FoundationGeo/data/blender \
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
