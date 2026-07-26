"""Configuration helpers for the desktop degradation editor."""

import copy
import tempfile
from pathlib import Path

import yaml


STAGE_LIBRARY = {
    "resize": {
        "title": "Resize",
        "description": "Reduce spatial detail before encoding.",
        "params": {"fixed_scale": 0.5},
        "enabled": True,
        "probability": 1.0,
    },
    "noise": {
        "title": "Noise",
        "description": "Add luminance and chrominance noise.",
        "params": {"y_strength_range": [1, 10], "uv_strength_range": [1, 10], "types": ["u", "t", "a"]},
        "enabled": False,
        "probability": 1.0,
    },
    "halo": {
        "title": "Sharpening halo",
        "description": "Apply a light unsharp-mask effect.",
        "params": {"luma_x_range": [3, 5], "luma_y_range": [3, 5], "luma_amount_range": [0, 0.5]},
        "enabled": False,
        "probability": 0.2,
    },
    "blur": {
        "title": "Blur",
        "description": "Apply a selected blur filter.",
        "params": {"enabled_types": ["gaussian"], "gaussian": {"sigma_range": [0.7, 2], "steps_range": [1, 3]}},
        "enabled": False,
        "probability": 0.4,
    },
    "ghosting": {
        "title": "Ghosting",
        "description": "Blend delayed, offset image ghosts.",
        "params": {"num_ghosts_range": [1, 2], "opacity_range": [0.05, 0.15], "delay_range": [1, 2], "offset_x_range": [-3, 3], "offset_y_range": [-2, 2], "enable_color_shift": True},
        "enabled": False,
        "probability": 0.1,
    },
    "codec": {
        "title": "Codec encode",
        "description": "Required final encoding step.",
        "params": {"h264": {"probability": 1.0, "quality_range": [23, 33]}},
        "enabled": True,
        "probability": 1.0,
    },
}

DEFAULT_STAGE_ORDER = ["resize", "noise", "halo", "blur", "ghosting", "codec"]


def default_stages():
    return [
        {"name": name, "enabled": STAGE_LIBRARY[name]["enabled"], "probability": STAGE_LIBRARY[name]["probability"]}
        for name in DEFAULT_STAGE_ORDER
    ]


def build_create_config(stages, base_config_path=None, chunking_strategy=None):
    """Build a version 2 config from GUI stages and an optional base config."""
    config = {"version": 2}
    if base_config_path:
        path = Path(base_config_path)
        if not path.is_file():
            raise ValueError(f"Configuration file not found: {path}")
        try:
            config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid YAML configuration: {error}") from error
        if not isinstance(config, dict):
            raise ValueError("Configuration root must be a mapping")
        config = copy.deepcopy(config)
        config.setdefault("version", 2)

    pipeline = []
    seen = set()
    for stage in stages:
        name = stage.get("name")
        if name not in STAGE_LIBRARY or name in seen:
            raise ValueError(f"Invalid degradation stage: {name!r}")
        seen.add(name)
        definition = STAGE_LIBRARY[name]
        enabled = bool(stage.get("enabled", definition["enabled"]))
        probability = stage.get("probability", definition["probability"])
        if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
            raise ValueError(f"{definition['title']} probability must be between 0 and 1")
        if name == "codec":
            if not enabled or probability != 1:
                raise ValueError("Codec encode must stay enabled with probability 1")
            continue
        pipeline.append({"name": name, "enabled": enabled, "probability": float(probability), "params": copy.deepcopy(definition["params"])})

    if "codec" not in seen:
        raise ValueError("Codec encode is required")
    codec = STAGE_LIBRARY["codec"]
    pipeline.append({"name": "codec", "enabled": True, "probability": 1.0, "params": copy.deepcopy(codec["params"])})
    create = config.setdefault("create", {})
    create["degradations"] = pipeline
    if chunking_strategy:
        create.setdefault("chunking", {})["strategy"] = chunking_strategy
    return config


def write_temp_create_config(stages, base_config_path=None, chunking_strategy=None):
    config = build_create_config(stages, base_config_path, chunking_strategy)
    with tempfile.NamedTemporaryFile(prefix="video-destroyer-gui-", suffix=".yaml", mode="w", encoding="utf-8", delete=False) as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
        return Path(handle.name)


def write_profile(path, stages, base_config_path=None, chunking_strategy=None):
    """Save the current pipeline as a reusable version 2 configuration."""
    config = build_create_config(stages, base_config_path, chunking_strategy)
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".yaml")
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path
