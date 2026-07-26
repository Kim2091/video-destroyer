"""Explicit-pair frame extraction with decoded-count verification."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from utils.scene_detector import SceneDetector

from .models import SequenceRecord


def _decode(video, destination, frame_format):
    destination.mkdir(parents=True, exist_ok=True)
    pattern = destination / f"frame_%08d.{frame_format}"
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video),
        "-vsync", "0", "-pix_fmt", "rgb24", str(pattern),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return sorted(destination.glob(f"frame_*.{frame_format}"))


def _starts(frame_count, config, video_path=None):
    length = config["sequence_length"]
    last = frame_count - length + 1
    if last <= 0:
        return []
    if config["mode"] == "scene_starts":
        if video_path is None:
            starts = [1]
        else:
            scenes = SceneDetector(config={"scene_detection": {}}).detect_scenes(str(video_path))
            starts = [scene[0].get_frames() + 1 for scene in scenes if scene[0].get_frames() + length <= frame_count]
    elif config["mode"] == "gapped":
        starts = list(range(1, last + 1, length + config.get("gap_frames", 0)))
    else:
        starts = list(range(1, last + 1, length))
    maximum = config.get("maximum_sequences_per_pair")
    if maximum is None or len(starts) <= maximum:
        return starts
    if maximum == 0:
        return []
    if maximum == 1:
        return starts[:1]
    step = (len(starts) - 1) / (maximum - 1)
    return [starts[int(index * step)] for index in range(maximum)]


def _sequence_id(pair_id, start, extract_config):
    fingerprint = json.dumps(extract_config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{pair_id}:{start}:{fingerprint}".encode("utf-8")).hexdigest()[:16]


def extract_pairs(records, resolve_paths, work_root, extract_config):
    work_root = Path(work_root)
    frames_root = work_root / "frames"
    format_name = extract_config["frame_format"].lower()
    results = []
    for pair in records:
        if pair.status != "validated":
            continue
        decode_root = work_root / "decoded" / pair.id
        pair_results = []
        written_names = []
        shutil.rmtree(decode_root, ignore_errors=True)
        try:
            hr_source, lr_source = resolve_paths(pair)
            hr_frames = _decode(hr_source, decode_root / "hr", format_name)
            lr_frames = _decode(lr_source, decode_root / "lr", format_name)
            if len(hr_frames) != len(lr_frames):
                raise ValueError(f"decoded frame counts differ: {len(hr_frames)} vs {len(lr_frames)}")
            for start in _starts(len(hr_frames), extract_config, hr_source):
                sequence_id = _sequence_id(pair.id, start, extract_config)
                names = [f"seq_{sequence_id}_Frame{index:05d}.{format_name}" for index in range(1, extract_config["sequence_length"] + 1)]
                staging = work_root / "staging" / sequence_id
                shutil.rmtree(staging, ignore_errors=True)
                (staging / "hr").mkdir(parents=True)
                (staging / "lr").mkdir(parents=True)
                for index, name in enumerate(names):
                    shutil.copyfile(hr_frames[start - 1 + index], staging / "hr" / name)
                    shutil.copyfile(lr_frames[start - 1 + index], staging / "lr" / name)
                for side in ("hr", "lr"):
                    target = frames_root / side
                    target.mkdir(parents=True, exist_ok=True)
                    for frame in (staging / side).iterdir():
                        frame.replace(target / frame.name)
                shutil.rmtree(staging)
                written_names.extend(names)
                pair_results.append(SequenceRecord(sequence_id, pair.id, start, len(names), names, list(names)))
            results.extend(pair_results)
        except Exception as error:
            # A rejected pair must not contribute a partial sequence to the run.
            for side in ("hr", "lr"):
                for name in written_names:
                    (frames_root / side / name).unlink(missing_ok=True)
            pair.status = "rejected"
            pair.rejection_reason = f"extraction: {error}"
        finally:
            shutil.rmtree(decode_root, ignore_errors=True)
    return results
