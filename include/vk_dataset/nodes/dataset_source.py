"""
Reading an existing dataset folder back into the graph, so the HR Process
nodes can run on a dataset that part 1 already wrote.

Image files become clips by decoding each frame with OpenCV inside a
VapourSynth frame callback. The ImageMagick reader plugin (imwri) would be
one line shorter, but the app does not ship it, and OpenCV is already needed
to write datasets; it also opens non-ASCII paths and treats PNG and JPEG the
same, so there is one code path and no extra plugin to install.
"""

import os

import cv2
import numpy as np
import vapoursynth as vs

from ..dataset_folders import image_map, select_hr_targets, sequence_groups, side_dir
from ..registry import Param, node

core = vs.core


def read_image(path):
    """An image file as an HxWx3 uint8 RGB array; alpha is dropped and 16-bit is reduced to 8-bit."""
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not read the image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def image_clip(paths, props=None):
    """
    The images as one RGB24 clip, a frame each in the order given. They must
    share one size, which the first image sets. `props` maps a frame prop name
    to one value per image.
    """
    if not paths:
        raise ValueError("an image clip needs at least one image")
    height, width = read_image(paths[0]).shape[:2]
    base = core.std.BlankClip(width=width, height=height, format=vs.RGB24, length=len(paths),
                              fpsnum=24, fpsden=1)
    props = props or {}

    def fill(n, f):
        image = read_image(paths[n])
        if image.shape[:2] != (height, width):
            raise ValueError(f"{paths[n]} is {image.shape[1]}x{image.shape[0]}, but the first image "
                             f"of its folder is {width}x{height}; process one image size at a time")
        out = f.copy()
        for plane in range(3):
            np.copyto(np.asarray(out[plane]), image[:, :, plane])
        for name, values in props.items():
            out.props[name] = values[n]
        return out

    return core.std.ModifyFrame(base, base, fill)


class MissingOutputs(dict):
    """A node result whose absent outputs raise a readable error when a connected node asks for them."""

    def __init__(self, values, reasons):
        super().__init__(values)
        self.reasons = reasons

    def __missing__(self, port):
        raise ValueError(self.reasons.get(port, f"the output {port!r} is not available"))


@node("load_dataset", category="Source", label="Load Dataset",
      description="Reads a dataset folder (hr/, lr/, show<N>_Frame<M> images). 'hr' holds one target per "
                  "sequence, 'lr' the matching centre LR frame of each, and 'lr_all' every LR frame of those "
                  "sequences. Frames carry their file stem in the VKName prop.",
      outputs={"hr": "clip", "lr": "clip", "lr_all": "clip"},
      params=[Param("directory", "path", "", help="The dataset folder that holds hr/ and lr/.")])
def load_dataset(ctx, *, directory):
    if not directory or not os.path.isdir(directory):
        raise FileNotFoundError(f"the dataset folder {directory!r} does not exist")
    hr_dir = side_dir(directory, "hr")
    if hr_dir is None:
        raise FileNotFoundError(f"{directory} has no hr folder")
    lr_dir = side_dir(directory, "lr")

    hr_images = image_map(hr_dir)
    if not hr_images:
        raise ValueError(f"no images found in {hr_dir}")
    lr_images = image_map(lr_dir) if lr_dir else {}
    targets = select_hr_targets(hr_images, lr_images)
    ctx.message(f"{len(targets)} HR targets of {len(hr_images)} images in {hr_dir}")

    index = {key: i for i, key in enumerate(targets)}
    hr_paths = [os.path.join(hr_dir, hr_images[key]) for key in targets]
    hr_names = [os.path.splitext(hr_images[key])[0] for key in targets]
    out = {"hr": image_clip(hr_paths, {"VKName": hr_names, "VKOwner": list(range(len(targets)))})}

    reasons = {}
    missing = [hr_images[k] for k in targets if k not in lr_images]
    if lr_dir is None:
        reasons["lr"] = reasons["lr_all"] = f"{directory} has no lr folder, which a connected node needs"
    elif missing:
        shown = ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
        reasons["lr"] = reasons["lr_all"] = (f"the lr folder has no match for {len(missing)} HR images "
                                             f"({shown}), which a connected node needs")
    else:
        lr_paths = [os.path.join(lr_dir, lr_images[k]) for k in targets]
        lr_names = [os.path.splitext(lr_images[k])[0] for k in targets]
        out["lr"] = image_clip(lr_paths, {"VKName": lr_names, "VKOwner": list(range(len(targets)))})

        # A sequence target owns every LR frame of its show; a flat image owns its one LR image.
        lr_groups = sequence_groups(lr_images)
        all_paths, all_names, owners = [], [], []
        for key in targets:
            keys = [key]
            for show, frames in sequence_groups({key: hr_images[key]}).items():
                keys = [frames_key for _, frames_key in sorted(lr_groups[show].items())]
            for frame_key in keys:
                all_paths.append(os.path.join(lr_dir, lr_images[frame_key]))
                all_names.append(os.path.splitext(lr_images[frame_key])[0])
                owners.append(index[key])
        out["lr_all"] = image_clip(all_paths, {"VKName": all_names, "VKOwner": owners})
    return MissingOutputs(out, reasons)
