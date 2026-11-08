"""
Building a denoising dataset from a video's static shots (VSR-Dataset-Creator's
"Denoise Dataset", see NOTICE.md).

A run of N consecutive frames with no scene movement shows the same picture N
times with different grain. Averaging the aligned frames gives a clean HR
target and the untouched centre frame is its noisy LR input. The nodes are
split so the model sits in the graph where the user can see and swap it:

    Find Static Runs -> Static Centres -> (a denoise model) -> Stack Static Runs
                                       \\-> the LR frames        -> Write Static Dataset

Finding runs tolerates gate weave but not local motion. The shift between
consecutive frames is measured by phase correlation and removed before the
frames are compared; the remaining difference is pooled into blocks and scored
as (largest block - median block), so grain, which raises every block alike,
cancels out, while a blink or a moving mouth spikes a few blocks.
"""

import json
import math
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
import vapoursynth as vs

from ..media import frame_array, to_gray8, to_rgb24
from ..registry import Param, node
from .hr_process import _copy_with, _to_rgb24, rgb_float

core = vs.core


@dataclass
class StaticRun:
    start_frame: int
    center_frame: int
    complexity: float
    mean_residual: float
    scene: int = 0
    exported: bool = True
    note: str = ""


@dataclass
class StaticRuns:
    window: int
    runs: list = field(default_factory=list)
    scenes_seen: int = 1
    capped_by_scene: int = 0
    video: dict = field(default_factory=dict)

    @property
    def exported(self):
        return [r for r in self.runs if r.exported]


def analysis_clip(clip, width):
    """The clip as full-range float luma no wider than `width`."""
    height = max(1, round(clip.height * min(width, clip.width) / clip.width))
    width = min(width, clip.width)
    if clip.format.color_family == vs.RGB:
        return core.resize.Bicubic(clip, width, height, format=vs.GRAYS, matrix_s="709")
    luma = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    return core.resize.Bicubic(luma, width, height, format=vs.GRAYS, range_s="full")


class StaticPairChecker:
    """Decides whether two consecutive analysis frames show zero scene movement."""

    def __init__(self, block_size, block_threshold, max_weave_px):
        self.block_size = block_size
        self.block_threshold = block_threshold
        self.max_weave_px = max_weave_px
        self._window = None

    def check(self, previous, current):
        """Returns (is_static, residual metric, weave in pixels)."""
        if self._window is None or self._window.shape != previous.shape:
            self._window = cv2.createHanningWindow((previous.shape[1], previous.shape[0]), cv2.CV_32F)
        # phaseCorrelate windows its inputs in place; copies keep the frames intact for the next pair.
        (dx, dy), _response = cv2.phaseCorrelate(previous.copy(), current.copy(), self._window)
        weave = math.hypot(dx, dy)
        if weave > self.max_weave_px:
            return False, np.inf, weave

        shifted = cv2.warpAffine(previous, np.float32([[1, 0, dx], [0, 1, dy]]),
                                 (previous.shape[1], previous.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        diff = np.abs(current - shifted)
        # The shift and the edges leave unreliable borders.
        margin = math.ceil(weave) + 4
        h, w = diff.shape
        size = self.block_size
        if h - 2 * margin < size * 2 or w - 2 * margin < size * 2:
            return False, np.inf, weave
        diff = diff[margin:h - margin, margin:w - margin]
        rows, cols = diff.shape[0] // size, diff.shape[1] // size
        blocks = diff[:rows * size, :cols * size].reshape(rows, size, cols, size).mean(axis=(1, 3))
        metric = float(blocks.max() - np.median(blocks))
        return metric <= self.block_threshold, metric, weave


def complexity_score(gray8):
    """Mean absolute Laplacian of the median-blurred luma; flat holds score near zero."""
    blurred = cv2.medianBlur(gray8, 5)
    return float(np.mean(np.abs(cv2.Laplacian(blurred.astype(np.float32) / 255.0, -1))))


@node("find_static_runs", category="Static Dataset", label="Find Static Runs",
      description="Scans the video for runs of consecutive frames with no scene movement, tolerating "
                  "gate weave, and keeps a limited number per scene.",
      inputs={"clip": "clip"}, outputs={"runs": "static_runs"},
      params=[
          Param("window", "int", 9, min=3, help="Frames per run; the middle one is the centre."),
          Param("max_pairs", "int", 0, min=0, help="Stop after this many runs (0 = scan the whole video)."),
          Param("analysis_width", "int", 640, min=64, help="The scan looks at the video shrunk to this width."),
          Param("block_size", "int", 16, min=2, help="Edge of the blocks the residual is pooled into."),
          Param("block_threshold", "float", 0.0075, min=0.0,
                help="Largest allowed (max block - median block) difference, in 0-1 luma."),
          Param("max_weave_px", "float", 2.5, min=0.0, help="Largest global drift per frame, in analysis pixels."),
          Param("min_gap", "int", 12, min=0, help="Frames between the starts of accepted runs."),
          Param("min_complexity", "float", 0.0, min=0.0, help="Skips flat or blank holds scoring below this."),
          Param("max_per_scene", "int", 3, min=0, help="Runs kept per scene, so a slow zoom cannot flood the "
                                                       "dataset with near-identical stacks (0 = no limit)."),
          Param("scene_cut_floor", "float", 0.08, min=0.0, help="Smallest mean luma difference that counts as a cut."),
          Param("scene_cut_ratio", "float", 2.5, min=1.0, help="A cut must spike this many times over the frame before."),
          Param("scene_cut_cooldown", "int", 12, min=0, help="Frames between detected cuts."),
      ])
def find_static_runs(ctx, clip, *, window, max_pairs, analysis_width, block_size, block_threshold, max_weave_px,
                     min_gap, min_complexity, max_per_scene, scene_cut_floor, scene_cut_ratio,
                     scene_cut_cooldown):
    window = max(3, int(window))
    centre_offset = window // 2
    checker = StaticPairChecker(block_size, block_threshold, max_weave_px)
    gray8 = to_gray8(clip)
    found = StaticRuns(window=window, video={
        "path": next(iter(ctx.video_paths.values()), None),
        "width": clip.width, "height": clip.height, "frames": clip.num_frames})

    previous = None
    static_pairs = 0
    captured = False
    last_start = -(10 ** 9)
    residuals = []
    scene = 0
    scene_exports = 0
    exported = 0
    recent_diffs = deque(maxlen=240)
    previous_diff = 0.0
    last_cut = -(10 ** 9)

    for n, frame in enumerate(analysis_clip(clip, analysis_width).frames(close=True)):
        if max_pairs > 0 and exported >= max_pairs:
            ctx.message("Reached the requested number of runs, stopping the scan")
            break
        if n % 100 == 0:
            ctx.emit({"message": f"{ctx.video_name}: scanning frame {n} of {clip.num_frames}"})
        current = cv2.GaussianBlur(np.array(frame[0], dtype=np.float32), (3, 3), 0.0)

        if previous is not None:
            # A cut is a one-frame spike: well above the rolling median and
            # MAD, and several times the previous frame's difference, so a
            # sustained pan does not retrigger it.
            diff = float(np.mean(np.abs(current - previous)))
            threshold = scene_cut_floor
            if len(recent_diffs) >= 24:
                values = np.asarray(recent_diffs)
                median = float(np.median(values))
                mad = float(np.median(np.abs(values - median)))
                threshold = max(threshold, median + 8.0 * max(mad * 1.4826, 1.0e-6))
            if (diff >= threshold and diff >= scene_cut_ratio * max(previous_diff, 1.0e-4)
                    and n - last_cut >= scene_cut_cooldown):
                scene += 1
                scene_exports = 0
                last_cut = n
            else:
                recent_diffs.append(diff)
            previous_diff = diff

            is_static, residual, _weave = checker.check(previous, current)
            if is_static:
                static_pairs += 1
                residuals.append(residual if np.isfinite(residual) else 0.0)
            else:
                static_pairs = 0
                captured = False
        previous = current

        if static_pairs >= window - 1 and not captured and n - (window - 1) - last_start >= min_gap:
            start = n - (window - 1)
            if max_per_scene > 0 and scene_exports >= max_per_scene:
                # The scene's budget is spent; the rest of the streak is skipped.
                found.capped_by_scene += 1
                captured = True
                continue
            centre = start + centre_offset
            run = StaticRun(start_frame=start, center_frame=centre, scene=scene,
                            complexity=complexity_score(frame_array(gray8.get_frame(centre))),
                            mean_residual=float(np.mean(residuals[-(window - 1):])) if residuals else 0.0)
            found.runs.append(run)
            if min_complexity > 0 and run.complexity < min_complexity:
                run.exported = False
                run.note = f"complexity {run.complexity:.4f} below threshold"
            else:
                exported += 1
                scene_exports += 1
            captured = True
            last_start = start

    found.scenes_seen = scene + 1
    ctx.message(f"{ctx.video_name}: {exported} static runs from {found.scenes_seen} scenes")
    return {"runs": found}


def _require_runs(runs):
    if not runs.exported:
        raise ValueError("no static runs were found; loosen the thresholds or try another video")
    return runs.exported


@node("static_centers", category="Static Dataset", label="Static Centres",
      description="The centre frame of each static run, untouched, one frame per run. These become the LR "
                  "images and are what a denoise model turns into the alignment reference.",
      inputs={"clip": "clip", "runs": "static_runs"}, outputs={"clip": "clip"})
def static_centers(ctx, clip, runs):
    rgb = to_rgb24(clip)
    return {"clip": core.std.Splice([rgb[r.center_frame] for r in _require_runs(runs)], mismatch=False)}


def zstack_average(frames, sigma_clip):
    """
    The mean over a stack of float frames. Pixels further than sigma_clip
    standard deviations from the per-pixel median are left out, which protects
    the average from motion the detector and the alignment missed. A
    sigma_clip of 0 is a plain mean.
    """
    stack = np.stack(frames, axis=0)
    if sigma_clip <= 0:
        return stack.mean(axis=0)
    median = np.median(stack, axis=0)
    deviation = stack.std(axis=0) + 1e-6
    keep = (np.abs(stack - median) <= sigma_clip * deviation).astype(np.float32)
    weight = keep.sum(axis=0)
    mean = (stack * keep).sum(axis=0) / np.maximum(weight, 1e-6)
    return np.where(weight > 0, mean, median)


def align_to(image, reference, window):
    """
    `image` shifted onto `reference`. The global translation between the two
    luma pictures is estimated by phase correlation and refined by ECC, which
    is accurate to a fraction of a pixel, then removed with one cubic resample
    of the full-resolution frame.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    (dx, dy), _response = cv2.phaseCorrelate(gray, gray_ref, window)
    warp = np.float32([[1, 0, dx], [0, 1, dy]])
    try:
        # The warp maps reference coordinates to image coordinates, hence WARP_INVERSE_MAP below.
        _, warp = cv2.findTransformECC(gray_ref, gray, np.float32([[1, 0, -dx], [0, 1, -dy]]),
                                       cv2.MOTION_TRANSLATION,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5), None, 5)
        flags = cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP
    except cv2.error:
        flags = cv2.INTER_CUBIC
    return cv2.warpAffine(image, warp, (image.shape[1], image.shape[0]), flags=flags,
                          borderMode=cv2.BORDER_REPLICATE)


@node("static_stack", category="Static Dataset", label="Stack Static Runs",
      description="Aligns each run's frames to its reference (the denoised centre) and averages them with "
                  "sigma clipping into the HR target.",
      inputs={"clip": "clip", "runs": "static_runs", "reference": "clip"}, optional_inputs=("reference",),
      outputs={"clip": "clip"},
      params=[Param("sigma_clip", "float", 2.5, min=0.0,
                    help="Deviations from the per-pixel median beyond which a sample is dropped (0 = plain mean).")])
def static_stack(ctx, clip, runs, reference=None, *, sigma_clip):
    exported = _require_runs(runs)
    rgb = rgb_float(clip)
    centre = runs.window // 2
    frames = [core.std.Splice([rgb[r.start_frame + j] for r in exported], mismatch=False)
              for j in range(runs.window)]
    if reference is not None:
        reference = rgb_float(reference)
        if (reference.width, reference.height, reference.num_frames) != (clip.width, clip.height, len(exported)):
            raise ValueError("the reference must have the video's size and one frame per static run")
    sources = frames + ([reference] if reference is not None else [])
    window_cache = {}

    def stack(n, f):
        images = [frame_array(frame) for frame in f]
        ref = images[-1] if reference is not None else images[centre]
        if window_cache.get("size") != ref.shape[:2]:
            window_cache["size"] = ref.shape[:2]
            window_cache["window"] = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
        aligned = [images[j] if j == centre else align_to(images[j], ref, window_cache["window"])
                   for j in range(runs.window)]
        result = zstack_average(aligned, sigma_clip)
        return _copy_with(f[0], [result[:, :, p] for p in range(3)])

    return {"clip": core.std.ModifyFrame(frames[centre], sources, stack)}


@node("write_static_dataset", category="Output", label="Write Static Dataset",
      description="Writes hr/static_<centre frame>.png and lr/static_<centre frame>.png for every run, "
                  "and static_manifest.json with the runs and settings.",
      inputs={"hr": "clip", "lr": "clip", "runs": "static_runs"},
      params=[Param("directory", "path", "static_dataset", help="The dataset folder."),
              Param("folders", "json", {"hr": "hr", "lr": "lr"}, help="Folder name for each side.")])
def write_static_dataset(ctx, hr, lr, runs, *, directory, folders):
    from .output import write_image
    exported = _require_runs(runs)
    if not (hr.num_frames == lr.num_frames == len(exported)):
        raise ValueError("the HR and LR clips need one frame per static run")
    state = ctx.state()
    if "runs" not in state:
        state.update(runs=[], files=[], started=time.time(), directory=directory)
        ctx.on_finish(_write_manifest(ctx.node.id))
    sides = {"hr": hr, "lr": lr}
    paths = {side: os.path.join(directory, folders.get(side, side)) for side in sides}
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    names = [f"static_{r.center_frame:06d}.png" for r in exported]
    if set(names) & set(state["files"]):
        raise ValueError("a second video produced static runs with the same centre frame numbers; "
                         "run the videos in separate jobs or folders")
    rgb = {side: _to_rgb24(clip) for side, clip in sides.items()}
    for side, clip in rgb.items():
        for name, frame in zip(names, clip.frames(close=True)):
            write_image(os.path.join(paths[side], name), frame_array(frame), "png", 0)
        ctx.emit({"message": f"wrote {len(names)} {side} images"})
    state["files"].extend(names)
    state["runs"].extend(asdict(r) for r in runs.runs)
    state["video"] = runs.video
    state["scenes_seen"] = runs.scenes_seen
    state["capped"] = runs.capped_by_scene
    state["record"] = ctx.settings_record()
    state["summary"] = {"pairs_exported": len(state["files"])}
    return {}


def _write_manifest(node_id):
    def finish(ctx):
        state = ctx.state(node_id)
        manifest = {
            "settings": state["record"], "video": state["video"],
            "runs_found": len(state["runs"]), "pairs_exported": len(state["files"]),
            "runs_capped_by_scene": state["capped"], "scenes_seen": state["scenes_seen"],
            "runs": state["runs"], "elapsed_sec": round(time.time() - state["started"], 1),
        }
        with open(os.path.join(state["directory"], "static_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    return finish
