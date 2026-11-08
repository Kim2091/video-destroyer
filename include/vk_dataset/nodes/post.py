"""
Post-processing of picked sequences, ported from video-destroyer's
post_processor: tiling and the filters that drop bad sequences. Every node
takes sequences and returns sequences; tiling replaces each sequence with
crops of it, the filters move sequences they reject into `Sequences.rejected`.
The writer can put those into a "bad" folder (its bad_directory setting).

The original moved image files between folders and then removed the LR
frames whose HR frames had gone. Here a sequence is HR and LR frames
together, so dropping it drops both and no sync step is needed. The filters
look at the frames of one clip (HR by default), through the sequence's crop.

Tiling cuts a grid, or picks the most detailed tiles the way pair mode
does, with pair_select.py's ports of VSR-Dataset-Creator's tile measures.
"""

import math
from collections import OrderedDict

import cv2
import numpy as np

from ..media import frame_array, to_rgb24
from ..registry import Param, node
from .pair_select import float_rgb, laplacian_complexity_map, motion_weights, select_tiles
from .sequences import Sequence

SOBEL_EDGE_LEVEL = 30
BLACK_BORDER_LEVEL = 10
BLACK_BORDER_LIMIT = 40.0


class GrayFrames:
    """Reads frames of a sequences value as 8-bit gray, as cv2 would load the written PNGs."""

    def __init__(self, sequences, clip_name, cache_size=64):
        if clip_name not in sequences.clips:
            raise ValueError(f"the sequences have no clip {clip_name!r}; they have {', '.join(sequences.clips)}")
        self.clip = to_rgb24(sequences.clips[clip_name])
        self.cache = OrderedDict()
        self.cache_size = cache_size

    def get(self, n, crop=None):
        if n not in self.cache:
            rgb = frame_array(self.clip.get_frame(n))
            self.cache[n] = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        gray = self.cache[n]
        if crop:
            left, top, width, height = crop
            gray = gray[top:top + height, left:left + width]
        return gray


PICKS = ["grid", "detail"]


def grid_positions(width, height, tile_width, tile_height):
    return [(x, y) for y in range(0, height - tile_height + 1, tile_height)
            for x in range(0, width - tile_width + 1, tile_width)]


def alignment_step(base, clips):
    """
    The step tile corners keep to, so a tile lands on whole pixels of every
    clip: the least common multiple of each smaller clip's whole size ratio.
    """
    step = 1
    for clip in clips:
        ratio = base.width // clip.width if clip.width else 0
        if ratio > 1 and (clip.width * ratio, clip.height * ratio) == (base.width, base.height):
            step = math.lcm(step, ratio)
    return step


def detailed_positions(rgb, item, reference, tile_width, tile_height, step, count, min_detail, edge_margin,
                       motion_bias):
    """
    The most detailed tiles of a sequence, best first, as (x, y, detail).
    Detail is measured on the reference clip's centre frame; motion is
    measured on the clip with the most frames in the sequence.
    """
    numbers = item.frames[reference]
    target = float_rgb(rgb[reference], numbers[len(numbers) // 2])
    moving = max(item.frames, key=lambda name: len(item.frames[name]))
    run = [float_rgb(rgb[moving], n) for n in item.frames[moving]]
    weights = motion_weights(run, target.shape[:2], motion_bias)
    return select_tiles(laplacian_complexity_map(target), tile_width, tile_height, step, count, min_detail,
                        edge_margin, weights)


@node("tile", category="Sequences", label="Tile Sequences",
      description="Cuts every sequence into tiles: on a grid, the way video-destroyer's tiling does, or where "
                  "the picture is most detailed, the way VSR-Dataset-Creator's does. Each tile becomes a "
                  "sequence of its own, the same crop on every frame. The tile size is for the HR clip; other "
                  "clips get the size scaled by their width and height ratio to HR. Grid tiles are shuffled "
                  "once per chunk from the seed.",
      inputs={"sequences": "sequences"}, outputs={"sequences": "sequences"},
      params=[
          Param("pick", "choice", "grid", choices=PICKS,
                help="'grid' cuts the whole frame; 'detail' takes the most detailed tiles, best first."),
          Param("tile_width", "int", 512, help="Tile width in HR pixels.", min=1),
          Param("tile_height", "int", 512, help="Tile height in HR pixels.", min=1),
          Param("max_tiles", "int", 0, help="Keep at most this many tiles per sequence (0 = every tile).", min=0),
          Param("min_detail", "float", 0.0015,
                help="For 'detail': least mean detail a tile needs. Content-dependent; around 0.01 culls harder.",
                min=0.0),
          Param("edge_margin", "int", 8, help="For 'detail': tiles keep this many HR pixels clear of the edge.",
                min=0),
          Param("motion_bias", "float", 1.0,
                help="For 'detail': how strongly tiles are drawn to where the sequence moves (0 = detail alone).",
                min=0.0),
          Param("reference", "str", "hr", help="The clip the tile size is given for."),
      ])
def tile(ctx, sequences, *, pick, tile_width, tile_height, max_tiles, min_detail, edge_margin, motion_bias,
         reference):
    if pick not in PICKS:
        raise ValueError(f"pick must be one of {', '.join(PICKS)}")
    if reference not in sequences.clips:
        raise ValueError(f"the sequences have no clip {reference!r}")
    base = sequences.clips[reference]
    ratios = {name: (clip.width / base.width, clip.height / base.height) for name, clip in sequences.clips.items()}
    grid = grid_positions(base.width, base.height, tile_width, tile_height)
    if pick == "grid":
        ctx.rng.shuffle(grid)
        if max_tiles:
            grid = grid[:max_tiles]
        ctx.note("tiles_per_sequence", len(grid))
    else:
        step = alignment_step(base, sequences.clips.values())
        if tile_width % step or tile_height % step:
            raise ValueError(f"the tile size must be a multiple of {step}, the HR/LR scale, so LR tiles line up")
        rgb = {name: to_rgb24(clip) for name, clip in sequences.clips.items()}
        count = max_tiles or len(grid)
    if not grid:
        ctx.message(f"{ctx.video_name}: frames are smaller than the {tile_width}x{tile_height} tile; "
                    "their sequences are dropped")
    items, rejected = [], list(sequences.rejected)
    for item in sequences.items:
        if item.crops:
            raise ValueError("tile takes whole-frame sequences; it cannot tile a sequence that is already cropped")
        if not grid:
            rejected.append(Sequence(item.frames, item.crops, {**item.info, "rejected": "smaller than tile"}))
            continue
        if pick == "grid":
            positions = [(x, y, None) for x, y in grid]
        else:
            positions = detailed_positions(rgb, item, reference, tile_width, tile_height, step, count, min_detail,
                                           edge_margin, motion_bias)
            if not positions:
                rejected.append(Sequence(item.frames, item.crops, {**item.info, "rejected": "no detailed tile"}))
        for index, (x, y, detail) in enumerate(positions):
            crops = {}
            for name, (rx, ry) in ratios.items():
                crops[name] = (round(x * rx), round(y * ry), int(tile_width * rx), int(tile_height * ry))
            info = {**item.info, "tile": [x, y], "tile_index": index}
            if detail is not None:
                info["detail"] = round(detail, 6)
            items.append(Sequence(dict(item.frames), crops, info))
    return {"sequences": type(sequences)(sequences.clips, items, rejected)}


def black_borders(gray):
    """Percentage of the frame that is black letterbox or pillarbox bars, and the bar sizes."""
    height, width = gray.shape
    rows = gray.mean(axis=1)
    columns = gray.mean(axis=0)

    def run(means, start, stop, step):
        size = 0
        for i in range(start, stop, step):
            if means[i] > BLACK_BORDER_LEVEL:
                break
            size = abs(i - start) + 1
        return size

    top = run(rows, 0, height // 2, 1)
    bottom = run(rows, height - 1, height // 2, -1)
    left = run(columns, 0, width // 2, 1)
    right = run(columns, width - 1, width // 2, -1)
    border_pixels = (top + bottom) * width + (left + right) * (height - top - bottom)
    return border_pixels / (height * width) * 100, (top, bottom, left, right)


def frame_detail(gray):
    """Edge density in percent and variance of the picture inside any black bars."""
    percentage, (top, bottom, left, right) = black_borders(gray)
    height, width = gray.shape
    content = gray[top:height - bottom, left:width - right] if percentage < 90 else gray
    if content.size < 100:
        content = gray
    sobel_x = cv2.Sobel(content, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(content, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    edge_density = np.count_nonzero(magnitude > SOBEL_EDGE_LEVEL) / content.size * 100
    return edge_density, float(np.var(content)), percentage


def is_blank(gray, edge_threshold, variance_threshold):
    edge_density, variance, border_percentage = frame_detail(gray)
    if border_percentage > BLACK_BORDER_LIMIT:
        return True
    return edge_density < edge_threshold and variance < variance_threshold


@node("reject_blank", category="Sequences", label="Reject Blank Sequences",
      description="Drops sequences with too many blank or low-detail frames: a frame is blank when more than "
                  "40% of it is black bars, or when both its edge density and its variance are below the "
                  "thresholds.",
      inputs={"sequences": "sequences"}, outputs={"sequences": "sequences"},
      params=[
          Param("min_blank_frames", "int", 2, help="Blank frames that make a sequence bad.", min=1),
          Param("edge_threshold", "float", 5.0, help="Minimum edge density in percent.", min=0.0),
          Param("variance_threshold", "float", 200.0, help="Minimum variance of the gray picture.", min=0.0),
          Param("clip", "str", "hr", help="The clip whose frames are judged."),
      ])
def reject_blank(ctx, sequences, *, min_blank_frames, edge_threshold, variance_threshold, clip):
    frames = GrayFrames(sequences, clip)

    def check(item):
        blank = sum(is_blank(frames.get(n, item.crops.get(clip)), edge_threshold, variance_threshold)
                    for n in item.frames[clip])
        return "blank" if blank >= min_blank_frames else None
    result = sequences.filtered(check)
    ctx.note("rejected", len(result.rejected) - len(sequences.rejected))
    return {"sequences": result}


def motion_score(first, second, threshold):
    """Percentage of pixels whose gray value differs by more than the threshold."""
    changed = cv2.absdiff(first, second) > threshold
    return float(np.count_nonzero(changed)) / changed.size * 100


@node("reject_motion", category="Sequences", label="Reject by Motion",
      description="Drops sequences whose consecutive frames differ too little or too much, measured as the "
                  "percentage of pixels that change by more than a threshold. A sequence with one frame of "
                  "the judged clip cannot be measured and is kept.",
      inputs={"sequences": "sequences"}, outputs={"sequences": "sequences"},
      params=[
          Param("min_motion", "float", 1.0, help="Minimum motion in percent (-1 to disable).", min=-1.0),
          Param("max_motion", "float", 15.0, help="Maximum motion in percent (-1 to disable).", min=-1.0),
          Param("threshold", "int", 30, help="Pixel difference that counts as changed (0-255).", min=0, max=255),
          Param("clip", "str", "hr", help="The clip whose frames are judged."),
      ])
def reject_motion(ctx, sequences, *, min_motion, max_motion, threshold, clip):
    frames = GrayFrames(sequences, clip)

    def check(item):
        numbers = item.frames[clip]
        crop = item.crops.get(clip)
        for a, b in zip(numbers, numbers[1:]):
            score = motion_score(frames.get(a, crop), frames.get(b, crop), threshold)
            if min_motion >= 0 and score < min_motion:
                return "too little motion"
            if max_motion >= 0 and score > max_motion:
                return "too much motion"
        return None
    result = sequences.filtered(check)
    ctx.note("rejected", len(result.rejected) - len(sequences.rejected))
    return {"sequences": result}


@node("reject_incomplete", category="Sequences", label="Reject Incomplete Sequences",
      description="Drops sequences that do not have the expected number of consecutive frames in every clip, "
                  "or whose frames run past the end of a clip. A clip with a single frame (the centre HR "
                  "frame) is accepted.",
      inputs={"sequences": "sequences"}, outputs={"sequences": "sequences"},
      params=[Param("length", "int", 5, help="Expected frames per sequence.", min=1)])
def reject_incomplete(ctx, sequences, *, length):
    def check(item):
        for name, numbers in item.frames.items():
            if len(numbers) == 1:
                complete = True
            else:
                complete = len(numbers) == length and numbers == list(range(numbers[0], numbers[0] + length))
            if not complete or numbers[-1] >= sequences.clips[name].num_frames or numbers[0] < 0:
                return "incomplete"
        return None
    result = sequences.filtered(check)
    ctx.note("rejected", len(result.rejected) - len(sequences.rejected))
    return {"sequences": result}
