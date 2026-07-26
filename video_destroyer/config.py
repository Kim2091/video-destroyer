"""Version 2 processing configuration loading and validation."""

import copy
from pathlib import Path

import yaml


class ConfigError(ValueError):
    pass


DEFAULT_CONFIG = {
    "version": 2,
    "seed": 1024,
    "create": {
        "chunking": {"strategy": "scene_detection", "minimum_seconds": 1.0},
        "degradations": [
            {"name": "resize", "probability": 1.0, "params": {"fixed_scale": 0.5}},
            {"name": "codec", "probability": 1.0, "params": {"h264": {"probability": 1.0, "quality_range": [23, 33]}}},
        ],
    },
    "extract": {
        "sequence_length": 5,
        "mode": "full_chunks",
        "frame_format": "png",
        "maximum_sequences_per_pair": None,
        "gap_frames": 0,
    },
    "curate": {
        "tiling": {"enabled": False, "width": 512, "height": 512},
        "blank_detection": {"enabled": False},
        "motion_detection": {"enabled": False},
    },
    "validation": {"expected_scale": None, "retain_rejected": False},
    "runtime": {"workers": None, "fail_on_rejection": False},
}

#: Degrade the supplied clips as they are, without splitting them first.
PRESPLIT_STRATEGY = "none"
CHUNK_STRATEGIES = {"scene_detection", "duration", "frame_count", PRESPLIT_STRATEGY}

_TOP_LEVEL = set(DEFAULT_CONFIG)
_SECTIONS = {
    "extract": {"sequence_length", "mode", "frame_format", "maximum_sequences_per_pair", "gap_frames"},
    "curate": {"tiling", "blank_detection", "motion_detection"},
    "validation": {"expected_scale", "retain_rejected"},
    "runtime": {"workers", "fail_on_rejection"},
    "create": {"chunking", "degradations"},
}


def _merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value


def _check_unknown(config):
    unknown = set(config) - _TOP_LEVEL
    if unknown:
        raise ConfigError("Unknown configuration key(s): " + ", ".join(sorted(unknown)))
    for section, allowed in _SECTIONS.items():
        value = config.get(section)
        if value is not None and not isinstance(value, dict):
            raise ConfigError(f"'{section}' must be a mapping")
        unknown = set(value or ()) - allowed
        if unknown:
            raise ConfigError(f"Unknown {section} key(s): " + ", ".join(sorted(unknown)))


def _validate(config, workflow):
    _check_unknown(config)
    if config.get("version") != 2:
        raise ConfigError("New commands require a version: 2 configuration")
    extract = config["extract"]
    if not isinstance(extract["sequence_length"], int) or extract["sequence_length"] <= 0:
        raise ConfigError("extract.sequence_length must be a positive integer")
    if extract["mode"] not in {"full_chunks", "scene_starts", "gapped"}:
        raise ConfigError("extract.mode must be full_chunks, scene_starts, or gapped")
    if extract["frame_format"].lower() not in {"png", "jpg", "jpeg"}:
        raise ConfigError("extract.frame_format must be png, jpg, or jpeg")
    maximum = extract["maximum_sequences_per_pair"]
    if maximum is not None and (not isinstance(maximum, int) or maximum < 0):
        raise ConfigError("extract.maximum_sequences_per_pair must be null or a non-negative integer")
    expected = config["validation"]["expected_scale"]
    if expected is not None and (not isinstance(expected, (int, float)) or expected <= 0):
        raise ConfigError("validation.expected_scale must be a positive number or null")
    if workflow == "create":
        create = config.get("create")
        if not isinstance(create, dict) or not isinstance(create.get("degradations"), list):
            raise ConfigError("create.degradations must be a list for the create workflow")
        if not create["degradations"]:
            raise ConfigError("create.degradations must include a codec degradation")
        if not any(item.get("name") == "codec" for item in create["degradations"] if isinstance(item, dict)):
            raise ConfigError("create.degradations must include a codec degradation")
        chunking = create.get("chunking") or {}
        if not isinstance(chunking, dict):
            raise ConfigError("create.chunking must be a mapping")
        strategy = chunking.get("strategy", "scene_detection")
        if strategy not in CHUNK_STRATEGIES:
            raise ConfigError("create.chunking.strategy must be one of: " + ", ".join(sorted(CHUNK_STRATEGIES)))


def load_config(path=None, workflow="import-pairs"):
    supplied = {}
    if path:
        config_path = Path(path)
        if not config_path.is_file():
            raise ConfigError(f"Configuration file not found: {config_path}")
        try:
            supplied = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as error:
            raise ConfigError(f"Invalid YAML configuration: {error}") from error
        if not isinstance(supplied, dict):
            raise ConfigError("Configuration root must be a mapping")
    _check_unknown(supplied)
    config = copy.deepcopy(DEFAULT_CONFIG)
    _merge(config, supplied)
    _validate(config, workflow)
    return config
