"""
Picking HR/LR training sequences out of an aligned pair, and dropping the
ones whose alignment failed, the way VSR-Dataset-Creator's "Extract
Sequences" does.

Pick Aligned Sequences takes, per chunk (a scene), the runs of frames with
the most motion: the LR run, and the single HR centre frame as its target.
Cut into Tiles (post.py) then cuts the most detailed crops of each run,
ranked by detail (median-blurred Laplacian, PepeDP's best-tile measure)
weighted by where the LR run moves. Drop Misaligned checks each crop three
ways:

* Timing. Every LR frame carries the LR frame number it came from (see
  align_time). Within a run those must follow on from the centre frame's;
  a run where several do not has drifted out of sync.
* Whole frame. The aligned HR centre frame, shrunk to LR size, must have
  edges where the LR centre frame has them (alignment_score).
* Per crop. The same check on the crop alone. The crops of a run that pass
  are kept best first, up to a count, so spare crops stand in for those
  that fail.

The crop, motion and alignment measures are modified ports of
VSR-Dataset-Creator's (Apache-2.0) tiling.py, part1.py and spatial.py,
itself after PepeDP (MIT); see NOTICE.md.
"""

from dataclasses import replace

import cv2
import numpy as np
import vapoursynth as vs

from ..media import frame_array, to_rgb24
from ..registry import Param, node
from .sequences import Sequence, Sequences

core = vs.core

MEDIAN_BLUR = 5
MOTION_SIZE = (32, 24)


def laplacian_complexity_map(rgb, median_blur=MEDIAN_BLUR):
    """Absolute Laplacian of the median-blurred grey image; rgb is float 0-1."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if median_blur > 1:
        size = median_blur | 1
        if size <= 5:
            gray = cv2.medianBlur(gray, size)
        else:
            gray = cv2.medianBlur((gray * 255).astype(np.uint8), size).astype(np.float32) / 255
    return np.abs(cv2.Laplacian(gray, -1))


def _best_tile_on_grid(ranking, tile_width, tile_height, step):
    """The tile with the highest sum whose corner sits on multiples of step, as (y, x)."""
    h, w = ranking.shape[:2]
    ys = np.arange(0, h - tile_height + 1, max(1, step), dtype=np.int32)
    xs = np.arange(0, w - tile_width + 1, max(1, step), dtype=np.int32)
    if len(ys) == 0 or len(xs) == 0:
        return 0, 0
    integral = cv2.integral(ranking.astype(np.float64, copy=False))
    sums = (integral[ys[:, None] + tile_height, xs[None, :] + tile_width]
            - integral[ys[:, None], xs[None, :] + tile_width]
            - integral[ys[:, None] + tile_height, xs[None, :]] + integral[ys[:, None], xs[None, :]])
    y, x = np.unravel_index(int(np.argmax(sums)), sums.shape)
    return int(ys[y]), int(xs[x])


def select_tiles(complexity, tile_width, tile_height, step, count, threshold, edge_margin, weights=None):
    """
    Up to `count` non-overlapping crops as (x, y, score), best first, with
    corners on multiples of step so the crops of a smaller clip line up
    exactly. Crops are ranked by complexity times weights; the score and the
    threshold use complexity alone, so the weighting moves crops without
    loosening the gate.
    """
    h, w = complexity.shape
    if h < tile_height or w < tile_width:
        return []
    ranking = complexity * weights if weights is not None else complexity.copy()
    regions = []
    for _ in range(count * 12):
        if len(regions) >= count:
            break
        y, x = _best_tile_on_grid(ranking, tile_width, tile_height, step)
        score = float(np.mean(complexity[y:y + tile_height, x:x + tile_width]))
        if score < threshold and weights is None:
            break
        near_edge = edge_margin and (x < edge_margin or y < edge_margin
                                     or x + tile_width > w - edge_margin or y + tile_height > h - edge_margin)
        if score >= threshold and not near_edge:
            regions.append((x, y, score))
        ranking[y:y + tile_height, x:x + tile_width] = -1.0
    return regions


def motion_weights(lr_run, hr_shape, bias):
    """
    1 + bias x where the LR run moves (mean absolute luma change, blurred,
    normalised to its 99.5th percentile), at HR size. None for a still run,
    so grain alone cannot steer the crops.
    """
    if bias <= 0 or len(lr_run) < 2:
        return None
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in lr_run]
    motion = np.mean([np.abs(b - a) for a, b in zip(grays, grays[1:])], axis=0)
    motion = cv2.GaussianBlur(motion, (0, 0), 2.0)
    top = float(np.quantile(motion, 0.995))
    if top <= 1e-6:
        return None
    motion = cv2.resize(np.clip(motion / top, 0.0, 1.0), (hr_shape[1], hr_shape[0]), interpolation=cv2.INTER_LINEAR)
    return (1.0 + float(bias) * motion).astype(np.float32)


def alignment_score(hr, lr):
    """
    Zero-mean correlation of gradient magnitudes between HR (shrunk to LR
    size) and LR, both float RGB 0-1. About 1 when their edges coincide;
    below about 0.3 the warp failed or the frames do not match.
    """
    h, w = lr.shape[:2]
    small = cv2.resize(hr, (w, h), interpolation=cv2.INTER_AREA)

    def gradient(img):
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32) / 255.0
        return cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

    a, b = gradient(small).reshape(-1), gradient(lr).reshape(-1)
    a, b = a - a.mean(), b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom >= 1e-8 else 0.0


def motion_profile(clip):
    """Mean absolute change of a 32x24 grey thumbnail between each frame and the next."""
    kwargs = {"matrix_s": "709"} if clip.format.color_family == vs.RGB else {}
    small = core.resize.Bilinear(clip, *MOTION_SIZE, format=vs.GRAYS, **kwargs)
    stats = core.std.PlaneStats(small[:-1], small[1:])
    return np.array([f.props["PlaneStatsDiff"] for f in stats.frames(close=True)], dtype=np.float32)


def pick_windows(motion, total, length, margin, count):
    """Starts of up to `count` non-overlapping runs, most motion first."""
    first, last = margin, total - margin - length
    if last < first:
        return []
    span = length - 1
    if span == 0:
        return [(first, 0.0)]
    csum = np.concatenate(([0.0], np.cumsum(motion, dtype=np.float64)))
    starts = np.arange(first, last + 1)
    scores = csum[starts + span] - csum[starts]
    picked = []
    for i in np.argsort(-scores, kind="stable"):
        start = int(starts[i])
        if all(abs(start - other) >= length for other, _ in picked):
            picked.append((start, float(scores[i])))
        if len(picked) >= count:
            break
    return picked


def timing_drift(sources, center, max_drift):
    """Frames of a run whose LR source number is off by more than max_drift, and the longest stretch of them."""
    expected = [sources[center] + i - center for i in range(len(sources))]
    off = [abs(s - e) > max_drift for s, e in zip(sources, expected)]
    longest = run = 0
    for item in off:
        run = run + 1 if item else 0
        longest = max(longest, run)
    return sum(off), longest


def float_rgb(clip, n):
    """Frame n of an RGB24 clip as float RGB 0-1."""
    return frame_array(clip.get_frame(n)).astype(np.float32) / 255.0


@node("pair_sequences", category="Sequences", label="Pick Aligned Sequences",
      description="Picks the runs with the most motion in each chunk of an aligned HR/LR pair: the LR run "
                  "plus the HR centre frame.",
      inputs={"hr": "clip", "lr": "clip"}, outputs={"sequences": "sequences"},
      params=[
          Param("length", "int", 9, help="LR frames per sequence; odd, so one is the centre.", min=1),
          Param("per_chunk", "int", 1, help="Runs picked per chunk.", min=1),
          Param("margin", "int", 4, help="Frames kept clear of the chunk's ends.", min=0),
      ])
def pair_sequences(ctx, hr, lr, *, length, per_chunk, margin):
    if length % 2 == 0:
        raise ValueError("the sequence length must be odd, so there is one centre frame")
    if hr.num_frames != lr.num_frames:
        raise ValueError(f"HR and LR differ in length ({hr.num_frames} and {lr.num_frames} frames); "
                         "align them in time first")
    scale = hr.width // lr.width
    if scale < 1 or (hr.width, hr.height) != (lr.width * scale, lr.height * scale):
        raise ValueError(f"HR is {hr.width}x{hr.height}, not a whole multiple of the LR's {lr.width}x{lr.height}; "
                         "align it in space first")

    center = length // 2
    items = []
    for start, motion in pick_windows(motion_profile(lr), lr.num_frames, length, margin, per_chunk):
        run = list(range(start, start + length))
        info = {"start": start, "motion": round(motion, 6)}
        sources = [lr.get_frame(n).props.get("LRFrame") for n in run]
        if all(s is not None for s in sources):
            info["lr_source_frames"] = [int(s) for s in sources]
        items.append(Sequence(frames={"hr": [run[center]], "lr": run}, info=info))
    return {"sequences": Sequences({"hr": hr, "lr": lr}, items)}


@node("reject_misaligned", category="Sequences", label="Drop Misaligned",
      description="Drops HR/LR sequences whose LR run drifted out of sync, whose centre frames do not line up, "
                  "or whose crop does not, then keeps the best crops left of each run.",
      inputs={"sequences": "sequences"}, outputs={"sequences": "sequences"},
      params=[
          Param("min_alignment_score", "float", 0.70, help="Whole-frame alignment gate.", min=0.0, max=1.0),
          Param("min_crop_alignment_score", "float", 0.82, help="Per-crop alignment gate (0 = none).",
                min=0.0, max=1.0),
          Param("max_drift", "int", 0, help="Frames an LR frame's source may be off its place in the run.", min=0),
          Param("drift_frames", "int", 2, help="Off frames that make a run drifted.", min=1),
          Param("crops_per_run", "int", 0,
                help="Crops kept per run, best first; the spares after them are dropped (0 = all).", min=0),
      ])
def reject_misaligned(ctx, sequences, *, min_alignment_score, min_crop_alignment_score, max_drift, drift_frames,
                      crops_per_run):
    for name in ("hr", "lr"):
        if name not in sequences.clips:
            raise ValueError(f"the sequences have no {name.upper()} clip; Drop Misaligned compares HR with LR")
    rgb = {name: to_rgb24(sequences.clips[name]) for name in ("hr", "lr")}
    frames, scores = {}, {}

    def frame(name, n):
        if (name, n) not in frames:
            frames[(name, n)] = float_rgb(rgb[name], n)
        return frames[(name, n)]

    def measure(item):
        """Why the item is dropped, or None, and the scores measured on the way."""
        hr_n = item.frames["hr"][len(item.frames["hr"]) // 2]
        run = item.frames["lr"]
        center = run.index(hr_n) if hr_n in run else len(run) // 2
        sources = item.info.get("lr_source_frames")
        if sources:
            count, longest = timing_drift(sources, center, max_drift)
            if count >= drift_frames and longest >= min(2, drift_frames):
                return "out of sync", {}
        key = (hr_n, run[center])
        if key not in scores:
            scores[key] = alignment_score(frame("hr", hr_n), frame("lr", run[center]))
        measured = {"align_score": round(scores[key], 4)}
        if scores[key] < min_alignment_score:
            return "misaligned", measured
        if min_crop_alignment_score > 0 and item.crops:
            hx, hy, hw, hh = item.crops["hr"]
            lx, ly, lw, lh = item.crops["lr"]
            crop_score = alignment_score(frame("hr", hr_n)[hy:hy + hh, hx:hx + hw],
                                         frame("lr", run[center])[ly:ly + lh, lx:lx + lw])
            measured["crop_align_score"] = round(crop_score, 4)
            if crop_score < min_crop_alignment_score:
                return "crop misaligned", measured
        return None, measured

    kept, rejected, per_run = [], list(sequences.rejected), {}
    for item in sequences.items:
        reason, measured = measure(item)
        info = {**item.info, **measured}
        if reason:
            rejected.append(replace(item, info={**info, "rejected": reason}))
            continue
        run = (tuple(item.frames["hr"]), tuple(item.frames["lr"]))
        per_run[run] = per_run.get(run, 0) + 1
        if not crops_per_run or per_run[run] <= crops_per_run:
            kept.append(replace(item, info=info))
    ctx.note("rejected", len(rejected) - len(sequences.rejected))
    return {"sequences": Sequences(sequences.clips, kept, rejected)}
