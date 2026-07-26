"""Create workflow adapter for the established split/degradation implementation."""

import hashlib
import random
import shutil
from pathlib import Path

from utils.codec_handler import CodecHandler
from utils.video_processor import VideoProcessor

from ..models import PairRecord
from ..pairing import VIDEO_EXTENSIONS, pair_id
from ..run_store import RunStore
from .common import run_dataset_workflow


def _sources(input_path):
    input_path = Path(input_path).resolve()
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported source video extension: {input_path.suffix}")
        return [(input_path.stem, input_path)]
    if not input_path.is_dir():
        raise ValueError(f"Source input does not exist: {input_path}")
    files = sorted(path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS and not any(part.startswith(".") for part in path.relative_to(input_path).parts))
    if not files:
        raise ValueError(f"No supported source videos found in {input_path}")
    return [(path.relative_to(input_path).with_suffix("").as_posix(), path) for path in files]


def _legacy_config(config, source, chunks_directory, log_directory):
    create = config["create"]
    chunking = create.get("chunking", {})
    degradations = []
    codecs = {}
    for definition in create["degradations"]:
        item = dict(definition)
        item.setdefault("enabled", True)
        item.setdefault("params", {})
        if item["name"] == "codec":
            normalized_params = {}
            for name, settings in item["params"].items():
                settings = dict(settings)
                quality = settings.get("quality_range", [23, 33])
                if isinstance(quality, list):
                    settings["quality_range"] = {"min": quality[0], "max": quality[1]}
                codecs[name] = settings
                normalized_params[name] = settings
            item["params"] = normalized_params
        degradations.append(item)
    if not codecs:
        raise ValueError("create.degradations must configure at least one codec")
    return {
        "input_video": str(source), "chunks_directory": str(chunks_directory),
        "chunk_strategy": chunking.get("strategy", "scene_detection"),
        "chunk_duration": chunking.get("duration_seconds", 10),
        "frames_per_chunk": chunking.get("frames", 300),
        "min_chunk_duration": chunking.get("minimum_seconds", 1.0),
        "scene_detection": chunking.get("scene_detection", {"strip_audio": True}),
        "degradations": degradations, "codecs": codecs,
        "logging": {"directory": str(log_directory), "filename": "run.log", "level": "INFO"},
    }


def discover_generated_pairs(store):
    """Regenerate only the application-owned create-stage outputs on recovery."""
    records = []
    config = store.run["config"]
    for source_key, source in _sources(store.run["inputs"]["source"]):
        source_id = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:12]
        random.seed(int(hashlib.sha256(f"{config['seed']}:{source_key}".encode("utf-8")).hexdigest(), 16))
        chunks = store.root / ".work" / "clips" / source_id
        shutil.rmtree(chunks, ignore_errors=True)
        legacy = _legacy_config(config, source, chunks, store.root / "logs")
        processor = VideoProcessor(legacy, CodecHandler(legacy["codecs"]))
        try:
            pairs = processor.process_video()
        finally:
            processor.logger.close()
        for index, (hr_path, lr_path) in enumerate(pairs, 1):
            key = f"{source_key}/chunk-{index:04d}"
            records.append(PairRecord(pair_id(key), key, "generated", "owned", Path(hr_path).relative_to(store.root).as_posix(), Path(lr_path).relative_to(store.root).as_posix()))
    return records


def start(input_path, output, config):
    inputs = {"source": str(Path(input_path).resolve())}
    store = RunStore.create(output, "create", config, inputs)

    return run_dataset_workflow(store, lambda: discover_generated_pairs(store))
