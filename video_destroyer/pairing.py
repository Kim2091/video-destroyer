"""Deterministic, source-independent HR/LR pair discovery."""

import hashlib
import os
import shutil
from pathlib import Path

from .models import PairRecord

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".m4v"}


class PairingError(ValueError):
    pass


def pair_id(key):
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _discover(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise PairingError(f"Input root is not a directory: {root}")
    found = {}
    folded = {}
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(name for name in directories if not name.startswith("."))
        for name in sorted(files):
            file_path = Path(current) / name
            if file_path.suffix.lower() not in VIDEO_EXTENSIONS or not file_path.is_file():
                continue
            key = file_path.relative_to(root).with_suffix("").as_posix()
            if key in found:
                raise PairingError(f"Duplicate pair key in {root}: {key}")
            collision = folded.get(key.casefold())
            if collision is not None:
                raise PairingError(f"Case-folded pair key collision in {root}: {collision} and {key}")
            found[key] = file_path
            folded[key.casefold()] = key
    return found


MATERIALIZE_MODES = ("copy", "hardlink")


def _materialize(source, destination, mode):
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Discovery re-runs when a run is resumed before it completed, so it can find
    # the clips an earlier attempt already placed here.
    destination.unlink(missing_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError as error:
        raise PairingError(f"Could not hardlink {source} to {destination}: {error}") from error


def discover_import_pairs(hr_root, lr_root, materialize=None, run_root=None):
    if materialize is not None and materialize not in MATERIALIZE_MODES:
        raise PairingError(f"Unknown materialization mode: {materialize}")
    hr_root, lr_root = Path(hr_root).resolve(), Path(lr_root).resolve()
    hr, lr = _discover(hr_root), _discover(lr_root)
    for folded_key in set(key.casefold() for key in hr) & set(key.casefold() for key in lr):
        hr_key = next(key for key in hr if key.casefold() == folded_key)
        lr_key = next(key for key in lr if key.casefold() == folded_key)
        if hr_key != lr_key:
            raise PairingError(f"Case-folded pair key collision between roots: {hr_key} and {lr_key}")
    unmatched_hr, unmatched_lr = sorted(set(hr) - set(lr)), sorted(set(lr) - set(hr))
    if unmatched_hr or unmatched_lr:
        parts = []
        if unmatched_hr:
            parts.append("HR only: " + ", ".join(unmatched_hr))
        if unmatched_lr:
            parts.append("LR only: " + ", ".join(unmatched_lr))
        raise PairingError("Unmatched pair keys; " + "; ".join(parts))
    records = []
    for key in sorted(hr):
        hr_path, lr_path = hr[key], lr[key]
        ownership = "referenced"
        hr_value, lr_value = hr_path.relative_to(hr_root).as_posix(), lr_path.relative_to(lr_root).as_posix()
        if materialize:
            if run_root is None:
                raise PairingError("A run root is required when materializing pairs")
            ownership = "owned"
            hr_destination = Path(run_root) / ".work" / "clips" / "hr" / hr_value
            lr_destination = Path(run_root) / ".work" / "clips" / "lr" / lr_value
            for source, destination in ((hr_path, hr_destination), (lr_path, lr_destination)):
                _materialize(source, destination, materialize)
            hr_value = hr_destination.relative_to(run_root).as_posix()
            lr_value = lr_destination.relative_to(run_root).as_posix()
        records.append(PairRecord(pair_id(key), key, "imported", ownership, hr_value, lr_value))
    return records
