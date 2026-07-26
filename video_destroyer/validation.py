"""Media and finalized-dataset validation."""

from fractions import Fraction
from pathlib import Path

import ffmpeg
from PIL import Image


def _rate(value):
    try:
        rate = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"Invalid frame rate: {value!r}")
    if rate <= 0:
        raise ValueError(f"Invalid frame rate: {value!r}")
    return rate


def probe_video(path):
    path = Path(path)
    if not path.is_file() or not path.stat().st_size:
        raise ValueError("file is missing, not regular, or empty")
    try:
        probe = ffmpeg.probe(str(path))
    except ffmpeg.Error as error:
        raise ValueError(f"FFmpeg could not probe file: {error}") from error
    streams = [stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"]
    if len(streams) != 1:
        raise ValueError(f"expected exactly one video stream, found {len(streams)}")
    stream = streams[0]
    try:
        duration = float(stream.get("duration") or probe.get("format", {}).get("duration"))
        info = {
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "pix_fmt": stream["pix_fmt"],
            "duration": duration,
            "fps": str(_rate(stream.get("r_frame_rate") or stream.get("avg_frame_rate"))),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"incomplete video metadata: {error}") from error
    if info["width"] <= 0 or info["height"] <= 0 or duration < 0:
        raise ValueError("invalid video dimensions or duration")
    return info


def validate_pairs(records, resolve_paths, expected_scale=None):
    """Preflight pairs, returning the accepted spatial scale and rejection count."""
    inferred = Fraction(str(expected_scale)) if expected_scale is not None else None
    rejected = 0
    for record in records:
        try:
            hr, lr = probe_video(resolve_paths(record)[0]), probe_video(resolve_paths(record)[1])
            hr_rate, lr_rate = _rate(hr["fps"]), _rate(lr["fps"])
            if hr_rate != lr_rate:
                raise ValueError(f"frame rates differ: {hr_rate} vs {lr_rate}")
            tolerance = max(Fraction(1, 1) / hr_rate, Fraction(1, 1) / lr_rate)
            if abs(Fraction(str(hr["duration"])) - Fraction(str(lr["duration"]))) > tolerance:
                raise ValueError("durations differ by more than one frame")
            scale = Fraction(lr["width"], hr["width"])
            height_scale = Fraction(lr["height"], hr["height"])
            if scale != height_scale:
                raise ValueError("HR and LR aspect ratios differ")
            if inferred is None:
                inferred = scale
            if scale != inferred:
                raise ValueError(f"scale differs from required {inferred}: {scale}")
            record.hr, record.lr, record.status, record.rejection_reason = hr, lr, "validated", None
        except Exception as error:
            record.status = "rejected"
            record.rejection_reason = str(error)
            rejected += 1
    return inferred, rejected


def validate_dataset(dataset_root, sequences, expected_frame_count, expected_scale=None):
    root = Path(dataset_root)
    hr_root, lr_root = root / "hr", root / "lr"
    if not hr_root.is_dir() or not lr_root.is_dir():
        return ["dataset HR and LR directories are required"]
    hr = {path.relative_to(hr_root).as_posix() for path in hr_root.rglob("*") if path.is_file()}
    lr = {path.relative_to(lr_root).as_posix() for path in lr_root.rglob("*") if path.is_file()}
    errors = []
    if not hr:
        errors.append("dataset contains no accepted frames")
    if hr != lr:
        errors.append("HR and LR dataset filenames differ")
    accepted = [record for record in sequences if record.status == "accepted"]
    if not accepted:
        errors.append("dataset contains no accepted sequences")
    expected_names = {name for record in accepted for name in record.hr_files}
    if hr != expected_names or lr != expected_names:
        errors.append("dataset files do not exactly match accepted sequence manifests")
    if len(hr) != sum(record.frame_count for record in accepted):
        errors.append("manifest frame count does not match dataset")
    scale = None if expected_scale is None else Fraction(str(expected_scale))
    for record in accepted:
        if record.frame_count != expected_frame_count or len(record.hr_files) != expected_frame_count or record.hr_files != record.lr_files:
            errors.append(f"sequence {record.id} has an invalid frame count")
            continue
        for filename in record.hr_files:
            hr_path, lr_path = hr_root / filename, lr_root / filename
            try:
                with Image.open(hr_path) as hr_image, Image.open(lr_path) as lr_image:
                    current_scale = Fraction(lr_image.width, hr_image.width)
                    if current_scale != Fraction(lr_image.height, hr_image.height):
                        errors.append(f"image aspect ratio mismatch: {filename}")
                    elif scale is None:
                        scale = current_scale
                    elif current_scale != scale:
                        errors.append(f"image scale mismatch: {filename}")
            except (OSError, ZeroDivisionError) as error:
                errors.append(f"unreadable image {filename}: {error}")
    return errors
