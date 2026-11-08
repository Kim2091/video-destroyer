"""
Writing a dataset to disk, in the layout traiNNer-redux reads for video
models: one folder per clip ("HR", "LR"), and per sequence the files
show<N>_Frame<M>.<ext> numbered from 1. M is the frame's place in the
sequence, so an HR folder that holds only each run's centre frame has
show<N>_Frame5 beside LR's show<N>_Frame1 to 9.

Numbering continues after the highest show number already in the folders,
so a second run adds to a dataset rather than overwriting it. Every
sequence's source video, chunk, frames, crops and drawn settings go into
manifest.json beside the folders.

Several writers can share one dataset folder, each in its own subfolder. A
writer with variants writes each sequence once per variant: the same HR
frames beside LR frames drawn anew, each variant a sequence of its own.
"""

import json
import os
import re

import cv2

from ..media import frame_array, to_rgb24
from ..registry import Param, node

SHOW_PATTERN = re.compile(r"^show(\d+)_Frame\d+\.", re.IGNORECASE)


def next_show_number(folders):
    highest = 0
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            match = SHOW_PATTERN.match(name)
            if match:
                highest = max(highest, int(match.group(1)))
    return highest + 1


def write_image(path, rgb, ext, quality):
    """Writes an RGB array. cv2.imwrite cannot open non-ASCII paths on Windows, so the bytes are written here."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)] if ext in ("jpg", "jpeg") else [cv2.IMWRITE_PNG_COMPRESSION, 3]
    ok, data = cv2.imencode("." + ext, bgr, params)
    if not ok:
        raise RuntimeError(f"could not encode {path}")
    with open(path, "wb") as f:
        f.write(data.tobytes())


@node("write_dataset", category="Output", label="Write Dataset", variants=True,
      description="Writes each sequence's frames as images, one folder per clip, plus manifest.json. "
                  "Sequences that filter nodes rejected are written the same way to the bad folder, if one is set.",
      inputs={"sequences": "sequences"},
      params=[
          Param("directory", "path", "frames", help="The dataset folder."),
          Param("subfolder", "str", "", help="A folder inside the dataset folder, so several writers can share it."),
          Param("folders", "json", {"hr": "HR", "lr": "LR"}, help="Folder name for each clip."),
          Param("format", "choice", "png", choices=["png", "jpg"]),
          Param("jpeg_quality", "int", 95, min=1, max=100),
          Param("naming", "choice", "padded", choices=["padded", "plain"],
                help="'padded' writes show00001_Frame00001, 'plain' writes show1_Frame1."),
          Param("bad_directory", "path", "",
                help="Where rejected sequences go, laid out like the dataset with its own manifest. "
                     "Empty drops them."),
      ])
def write_dataset(ctx, sequences, *, directory, subfolder, folders, format, jpeg_quality, naming, bad_directory):
    state = ctx.state()
    if "targets" not in state:
        state["targets"] = {}
        ctx.on_finish(_write_manifests(ctx.node.id))
    rgb = {name: to_rgb24(clip) for name, clip in sequences.clips.items()}
    record = ctx.settings_record()
    summary = state.setdefault("summary", {})
    inside = (lambda path: os.path.join(path, subfolder)) if subfolder else (lambda path: path)
    jobs = [("sequences", inside(directory), sequences.items)]
    if bad_directory:
        jobs.append(("rejected", inside(bad_directory), sequences.rejected))
    for key, target_directory, items in jobs:
        target = _target(state, target_directory, sequences.clips, folders)
        _write_items(ctx, target, rgb, items, record, format, jpeg_quality, naming)
        summary[key] = summary.get(key, 0) + len(items)
    return {}


def _target(state, directory, clips, folders):
    """The folders, show counter and manifest entries of one output directory."""
    if directory not in state["targets"]:
        paths = {name: os.path.join(directory, folders.get(name, name.upper())) for name in clips}
        for path in paths.values():
            os.makedirs(path, exist_ok=True)
        state["targets"][directory] = {"paths": paths, "next_show": next_show_number(paths.values()), "entries": []}
    return state["targets"][directory]


def _write_items(ctx, target, rgb, items, record, format, jpeg_quality, naming):
    width = 5 if naming == "padded" else 0
    for item in items:
        show = target["next_show"]
        # Files are numbered by the frame's place in the whole run, so a clip
        # that holds only the centre frame names it as the centre frame.
        run = sorted(set().union(*item.frames.values()))
        for name, numbers in item.frames.items():
            crop = item.crops.get(name)
            for n in numbers:
                index = run.index(n) + 1
                image = frame_array(rgb[name].get_frame(n))
                if crop:
                    left, top, w, h = crop
                    image = image[top:top + h, left:left + w]
                file_name = f"show{show:0{width}d}_Frame{index:0{width}d}.{format}"
                write_image(os.path.join(target["paths"][name], file_name), image, format, jpeg_quality)
        start = ctx.chunk_range[0] if ctx.chunk_range else 0
        target["entries"].append({
            "show": show, "video": ctx.video_name, "chunk": ctx.chunk_index,
            "frames": {name: [start + n for n in numbers] for name, numbers in item.frames.items()},
            "crops": item.crops or None, "info": item.info or None, "settings": record,
            **({"variant": ctx.variant} if ctx.variants > 1 else {}),
        })
        target["next_show"] += 1


def _write_manifests(node_id):
    def finish(ctx):
        for directory, target in ctx.state(node_id)["targets"].items():
            path = os.path.join(directory, "manifest.json")
            existing = []
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    existing = json.load(f).get("sequences", [])
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"seed": ctx.seed, "sequences": existing + target["entries"]}, f, indent=1)
    return finish
