"""Sequence-level curation that never mutates source frame output in place."""

import hashlib
import shutil
from pathlib import Path

from PIL import Image, ImageChops, ImageStat


def _reject_for_blank(paths):
    for path in paths:
        with Image.open(path).convert("L") as image:
            if ImageStat.Stat(image).var[0] < 1.0:
                return "blank or low-detail frame"
    return None


def _reject_for_motion(paths):
    if len(paths) < 2:
        return None
    for first, second in zip(paths, paths[1:]):
        with Image.open(first).convert("L") as first_image, Image.open(second).convert("L") as second_image:
            if ImageStat.Stat(ImageChops.difference(first_image, second_image)).mean[0] == 0:
                return "no motion between consecutive frames"
    return None


def _copy_or_tile(source, target, names, tile_config):
    enabled = tile_config.get("enabled", False)
    for name in names:
        source_path, target_path = source / name, target / name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if not enabled:
            shutil.copyfile(source_path, target_path)
            continue
        with Image.open(source_path) as image:
            width, height = tile_config["width"], tile_config["height"]
            if image.width < width or image.height < height:
                raise ValueError("frame is smaller than configured tile")
            digest = hashlib.sha256(name.encode("utf-8")).digest()
            x = int.from_bytes(digest[:4], "big") % (image.width - width + 1)
            y = int.from_bytes(digest[4:8], "big") % (image.height - height + 1)
            image.crop((x, y, x + width, y + height)).save(target_path)


def _copy_tiled_pairs(hr_source, lr_source, hr_target, lr_target, names, tile_config):
    for name in names:
        with Image.open(hr_source / name) as hr_image, Image.open(lr_source / name) as lr_image:
            width_scale = lr_image.width / hr_image.width
            height_scale = lr_image.height / hr_image.height
            if width_scale != height_scale:
                raise ValueError("HR and LR frame aspect ratios differ")
            width, height = tile_config["width"], tile_config["height"]
            lr_width, lr_height = round(width * width_scale), round(height * height_scale)
            if hr_image.width < width or hr_image.height < height or lr_image.width < lr_width or lr_image.height < lr_height:
                raise ValueError("frame is smaller than configured tile")
            digest = hashlib.sha256(name.encode("utf-8")).digest()
            x = int.from_bytes(digest[:4], "big") % (hr_image.width - width + 1)
            y = int.from_bytes(digest[4:8], "big") % (hr_image.height - height + 1)
            hr_path, lr_path = hr_target / name, lr_target / name
            hr_path.parent.mkdir(parents=True, exist_ok=True)
            lr_path.parent.mkdir(parents=True, exist_ok=True)
            hr_image.crop((x, y, x + width, y + height)).save(hr_path)
            lr_x, lr_y = round(x * width_scale), round(y * height_scale)
            lr_image.crop((lr_x, lr_y, lr_x + lr_width, lr_y + lr_height)).save(lr_path)


def curate_sequences(sequences, work_root, config, retain_rejected, rejected_root):
    work_root, rejected_root = Path(work_root), Path(rejected_root)
    source_hr, source_lr = work_root / "frames" / "hr", work_root / "frames" / "lr"
    accepted_hr, accepted_lr = work_root / "accepted" / "hr", work_root / "accepted" / "lr"
    shutil.rmtree(accepted_hr.parent, ignore_errors=True)
    accepted_hr.mkdir(parents=True)
    accepted_lr.mkdir(parents=True)
    for sequence in sequences:
        if sequence.status == "rejected":
            continue
        hr_paths = [source_hr / name for name in sequence.hr_files]
        reason = None
        if config["blank_detection"].get("enabled"):
            reason = _reject_for_blank(hr_paths)
        if reason is None and config["motion_detection"].get("enabled"):
            reason = _reject_for_motion(hr_paths)
        try:
            if reason is None:
                if config["tiling"].get("enabled"):
                    _copy_tiled_pairs(source_hr, source_lr, accepted_hr, accepted_lr, sequence.hr_files, config["tiling"])
                else:
                    _copy_or_tile(source_hr, accepted_hr, sequence.hr_files, config["tiling"])
                    _copy_or_tile(source_lr, accepted_lr, sequence.lr_files, config["tiling"])
        except (OSError, ValueError) as error:
            reason = f"curation: {error}"
        if reason:
            # A failed copy or tile may have created one side before the error.
            for name in sequence.hr_files:
                (accepted_hr / name).unlink(missing_ok=True)
            for name in sequence.lr_files:
                (accepted_lr / name).unlink(missing_ok=True)
            sequence.status, sequence.rejection_reason = "rejected", reason
            if retain_rejected:
                _copy_or_tile(source_hr, rejected_root / "hr", sequence.hr_files, {"enabled": False})
                _copy_or_tile(source_lr, rejected_root / "lr", sequence.lr_files, {"enabled": False})
        else:
            sequence.status, sequence.rejection_reason = "accepted", None
    return sequences
