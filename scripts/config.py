#!/usr/bin/env python3
"""YAML 配置加载与合并"""

from __future__ import annotations


def load_yaml_config(path: str) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("缺少 PyYAML，请先安装：pip install pyyaml") from exc

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("YAML 顶层必须是字典")
    return data


def merge_render_config(args, config: dict):
    if not config:
        return args

    def pick(value, default):
        return value if value is not None else default

    args.blend_file = pick(getattr(args, "blend_file", None), config.get("input"))
    args.output = pick(getattr(args, "output", None), config.get("output"))
    args.camera = pick(getattr(args, "camera", None), config.get("camera"))
    args.width = pick(getattr(args, "width", None), config.get("width"))
    args.height = pick(getattr(args, "height", None), config.get("height"))

    if not args.export_animation and config.get("export_animation") is True:
        args.export_animation = True
    if args.frame_start is None:
        args.frame_start = config.get("frame_start")
    if args.frame_end is None:
        args.frame_end = config.get("frame_end")
    if args.frame_step == 1 and config.get("frame_step") not in (None, 1):
        args.frame_step = config.get("frame_step")

    if not args.skip_conversion and config.get("skip_conversion") is True:
        args.skip_conversion = True
    if args.colormap == "turbo" and config.get("colormap"):
        args.colormap = config.get("colormap")
    if args.blender is None:
        args.blender = config.get("blender")

    return args
