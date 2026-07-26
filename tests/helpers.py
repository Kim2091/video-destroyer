import importlib.util
import subprocess
import sys
from pathlib import Path

import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def make_video(path: Path, duration: float = 1.0, rate: int = 10, size: str = "64x48") -> Path:
    """Create a small, real CFR video fixture using FFmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}:duration={duration}",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def make_cut_video(path: Path) -> Path:
    """Create a real two-scene video with a hard color cut."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=black:size=64x48:rate=10:duration=2",
            "-f", "lavfi", "-i", "color=white:size=64x48:rate=10:duration=2",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0", "-an",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def write_image(path: Path, color, size=(64, 48)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def write_sequence(directory: Path, show: int, colors, prefix="show", size=(64, 48)):
    paths = []
    for index, color in enumerate(colors, 1):
        paths.append(write_image(directory / f"{prefix}{show:05d}_Frame{index:05d}.png", color, size))
    return paths


def minimal_config(tmp_path: Path, input_path: Path = None, chunks_name="chunks"):
    """Return a complete, fast configuration for integration tests."""
    config = {
        "input": str(input_path) if input_path else "",
        "chunks_directory": str(tmp_path / chunks_name),
        "chunk_strategy": "frame_count",
        "frames_per_chunk": 5,
        "chunk_duration": 1,
        "min_chunk_duration": 0.1,
        "scene_detection": {"threshold": 10, "downscale_factor": 1, "strip_audio": True, "max_scenes": 0},
        "logging": {"directory": str(tmp_path / "logs"), "filename": "test.log", "level": "WARNING"},
        "degradations": [
            {
                "name": "resize", "enabled": True, "probability": 1.0,
                "params": {"fixed_scale": 0.5, "scaling_filters": ["bilinear"], "down_up": {"enabled": False}},
            },
            {
                "name": "codec", "enabled": True, "probability": 1.0,
                "params": {"h264": {"probability": 1.0, "quality_range": [28, 28], "presets": ["ultrafast"]}},
            },
        ],
        "frame_extraction": {
            "auto_extract_frames": False, "output_directory": str(tmp_path / "frames"),
            "sequence_length": 2, "use_scene_detection": False, "extract_full_chunks": True,
            "time_gap": 0, "frame_skip": 0, "max_sequences_per_chunk": None,
            "skip_existing": False, "frame_format": "png", "verbose_logging": False,
        },
        "post_processing": {"enabled": False},
    }
    return config


def write_yaml(path: Path, config) -> Path:
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def load_tool_module(relative_path: str, module_name: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
