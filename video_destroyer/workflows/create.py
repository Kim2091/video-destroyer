"""Create workflow adapter for the established split/degradation implementation."""

import hashlib
import random
import shutil
from pathlib import Path

from utils.codec_handler import CodecHandler
from utils.video_processor import VideoProcessor

from ..config import PRESPLIT_STRATEGY
from ..models import PairRecord
from ..pairing import VIDEO_EXTENSIONS, pair_id
from ..run_store import RunStore
from .common import run_dataset_workflow


def _video_files(root):
    return sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
    )


def _keyed(paths, root, label):
    """Key each path by its relative path without the extension, rejecting collisions."""
    keyed = []
    seen = {}
    for path in paths:
        key = path.relative_to(root).with_suffix("").as_posix()
        if key in seen:
            raise ValueError(f"Duplicate {label} key {key!r}: {seen[key].name} and {path.name} differ only by extension")
        seen[key] = path
        keyed.append((key, path))
    return keyed


def _sources(input_path):
    input_path = Path(input_path).resolve()
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported source video extension: {input_path.suffix}")
        return [(input_path.stem, input_path)]
    if not input_path.is_dir():
        raise ValueError(f"Source input does not exist: {input_path}")
    files = _video_files(input_path)
    if not files:
        raise ValueError(f"No supported source videos found in {input_path}")
    return _keyed(files, input_path, "source video")


def _presplit_clips(input_path):
    """Return the clip root and its clips, which are degraded as supplied."""
    root = Path(input_path).resolve()
    if root.is_file():
        if root.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported clip extension: {root.suffix}")
        return root.parent, [(root.stem, root)]
    if not root.is_dir():
        raise ValueError(f"Source input does not exist: {root}")
    clips = _video_files(root)
    if not clips:
        raise ValueError(f"No supported clips found in {root}")
    return root, _keyed(clips, root, "clip")


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


def _seed(config, key):
    random.seed(int(hashlib.sha256(f"{config['seed']}:{key}".encode("utf-8")).hexdigest(), 16))


def _degrade(legacy_config, chunk_pairs):
    processor = VideoProcessor(legacy_config, CodecHandler(legacy_config["codecs"]))
    try:
        processor.process_chunks(chunk_pairs)
    finally:
        processor.logger.close()


def discover_presplit_pairs(store):
    """Degrade clips that are already split, without splitting them again."""
    config = store.run["config"]
    inputs = store.run["inputs"]
    clip_root, clips = _presplit_clips(inputs["source"])
    lr_root = Path(inputs["lr_root"])
    shutil.rmtree(lr_root, ignore_errors=True)

    records = []
    chunk_pairs = []
    for key, clip in clips:
        # Matroska accepts every codec this pipeline emits, whatever the source container was.
        lr_path = lr_root / f"{key}.mkv"
        lr_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_pairs.append((str(clip), str(lr_path)))
        records.append(PairRecord(
            pair_id(key), key, "generated", "referenced",
            clip.relative_to(clip_root).as_posix(), lr_path.relative_to(lr_root).as_posix(),
        ))

    _seed(config, clip_root.as_posix())
    # The first clip only supplies the media probe; no splitting is performed.
    _degrade(_legacy_config(config, clips[0][1], store.root / ".work" / "clips" / "presplit", store.root / "logs"), chunk_pairs)
    return records


def discover_generated_pairs(store):
    """Regenerate only the application-owned create-stage outputs on recovery."""
    records = []
    config = store.run["config"]
    for source_key, source in _sources(store.run["inputs"]["source"]):
        source_id = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:12]
        _seed(config, source_key)
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


def is_presplit(config):
    return (config.get("create") or {}).get("chunking", {}).get("strategy") == PRESPLIT_STRATEGY


def discover(store):
    return discover_presplit_pairs(store) if is_presplit(store.run["config"]) else discover_generated_pairs(store)


def start(input_path, output, config):
    source = Path(input_path).resolve()
    inputs = {"source": str(source)}
    if is_presplit(config):
        # HR is the supplied clips, read in place; only the LR side is run-owned.
        inputs["hr_root"] = str(source if source.is_dir() else source.parent)
        inputs["lr_root"] = str(Path(output).resolve() / ".work" / "clips" / "presplit" / "lr")
    store = RunStore.create(output, "create", config, inputs)

    return run_dataset_workflow(store, lambda: discover(store))
