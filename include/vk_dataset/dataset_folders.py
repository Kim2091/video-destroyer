"""
Reading the layout of a dataset folder: the hr and lr image folders and the
show<N>_Frame<M> sequences in them.

Part 1 writes one folder per clip, and per sequence one file per LR frame but
only the centre HR frame. Older datasets hold an HR file for every LR frame.
Both are told apart here, so a node that processes HR targets sees exactly
one target per sequence.
"""

import os
import re

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")
SHOW_FRAME_STEM = re.compile(r"^show(\d+)_Frame(\d+)$", re.IGNORECASE)


def side_dir(root, side):
    """The folder of a dataset side ("hr" or "lr"), matched without regard to case."""
    exact = os.path.join(root, side)
    if os.path.isdir(exact):
        return exact
    for name in sorted(os.listdir(root)):
        if name.lower() == side and os.path.isdir(os.path.join(root, name)):
            return os.path.join(root, name)
    return None


def image_map(folder):
    """Image files of a folder keyed by lower-cased stem, in natural name order."""
    images = {}
    for name in sorted(os.listdir(folder), key=str.casefold):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in IMAGE_EXTENSIONS or not os.path.isfile(os.path.join(folder, name)):
            continue
        key = stem.casefold()
        if key in images:
            raise ValueError(f"two images in {folder} share the stem {stem!r}: {images[key]} and {name}")
        images[key] = name
    return images


def sequence_groups(images):
    """{show number: {frame number: stem key}} for the show<N>_Frame<M> images of an image_map."""
    groups = {}
    for key, name in images.items():
        match = SHOW_FRAME_STEM.match(os.path.splitext(name)[0])
        if match:
            groups.setdefault(int(match.group(1)), {})[int(match.group(2))] = key
    return groups


def select_hr_targets(hr_images, lr_images=None):
    """
    The HR images to process, as a list of stem keys: one centre frame per
    sequence, and every flat image. The LR frames of a sequence name its
    centre when there are any; otherwise the HR frames do.
    """
    hr_groups = sequence_groups(hr_images)
    lr_groups = sequence_groups(lr_images or {})
    in_sequence = {key for frames in hr_groups.values() for key in frames.values()}
    selected = set(hr_images) - in_sequence
    for show, hr_frames in hr_groups.items():
        reference = sorted(lr_groups.get(show, hr_frames))
        source = "LR" if show in lr_groups else "HR"
        if len(reference) % 2 == 0:
            raise ValueError(f"sequence show{show} has {len(reference)} {source} frames; "
                             "a centre frame needs an odd length")
        centre = reference[len(reference) // 2]
        if centre not in hr_frames:
            have = ", ".join(f"Frame{n}" for n in sorted(hr_frames))
            raise ValueError(f"sequence show{show} has no HR image for its centre frame "
                             f"(Frame{centre}); it has {have}")
        selected.add(hr_frames[centre])
    return [key for key in hr_images if key in selected]
