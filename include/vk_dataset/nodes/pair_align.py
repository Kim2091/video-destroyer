"""
Lining up two releases of the same footage, for HR/LR pair datasets.

An LR release and an HR release rarely start on the same frame, may run at
slightly different speeds, and are framed a little differently. Pair mode
fixes that in two steps, the way VSR-Dataset-Creator's "Extract Sequences"
does:

* Time. align_videos.py finds the speed and per-section offsets between the
  releases, which maps every HR frame to roughly the right LR frame. Then
  vs_align.temporal picks the exact LR frame within a few frames of that
  guess. The result is an LR clip with one frame per HR frame, so the chunk
  node can cut both at the same frames.

* Space. vs_align.spatial (XFeat homography, then RIFE flow) warps HR onto
  the LR picture, upscaled to the training scale. The LR frames stay as they
  are and HR becomes the target that matches them pixel for pixel.

Both run on CUDA when PyTorch sees a GPU and on the CPU otherwise, which
works but is much slower.
"""

import contextlib
import os
import re
import sys

import vapoursynth as vs

from ..registry import Param, node

core = vs.core

# zimg's bicubic with b=0, c=0.75 is OpenCV's INTER_CUBIC, which
# VSR-Dataset-Creator uses for every rescale.
CUBIC = {"filter_param_a": 0.0, "filter_param_b": 0.75}

# The size frames are compared at when matching time.
MATCH_WIDTH = 320

BACKENDS = ["auto", "cuda", "cpu"]


def resolve_backend(backend):
    """'auto' is cuda when PyTorch has a GPU, else cpu."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; known: {', '.join(BACKENDS)}")
    if backend != "auto":
        return backend
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def to_rgbs(clip):
    if clip.format.id == vs.RGBS:
        return clip
    kwargs = {"matrix_in_s": "709"} if clip.format.color_family == vs.YUV else {}
    return core.resize.Point(clip, format=vs.RGBS, **kwargs)


def _align_videos():
    """align_videos.py lives beside the vk_dataset package, in include/."""
    include = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if include not in sys.path:
        sys.path.insert(0, include)
    import align_videos
    return align_videos


@contextlib.contextmanager
def _quiet_align(module, ctx, device):
    """
    Runs align_videos on the given torch device, with its progress lines
    turned into this run's messages: it prints its own protocol to stdout,
    which would otherwise be read as the dataset run's progress.
    """
    import torch
    emit, old_device = module.emit, module.DEVICE
    last = [None]

    def forward(**fields):
        # One message per stage: "Matching 3 of 40 moments" and the like
        # repeat for every sample.
        stage = re.sub(r"\d+", "#", fields.get("message", ""))
        if stage and stage != last[0]:
            last[0] = stage
            ctx.message(f"{ctx.video_name}: {fields['message']}")

    module.emit, module.DEVICE = forward, torch.device(device)
    try:
        yield
    finally:
        module.emit, module.DEVICE = emit, old_device


@node("crop_rescale", category="Transform", label="Crop and Rescale",
      description="Trims the frame's edges, then rescales it with a cubic filter.",
      inputs={"clip": "clip"}, outputs={"clip": "clip"},
      params=[
          Param("left", "int", 0, min=0),
          Param("right", "int", 0, min=0),
          Param("top", "int", 0, min=0),
          Param("bottom", "int", 0, min=0),
          Param("rescale", "float", 1.0, help="Size after cropping, as a fraction of the cropped size.", min=0.01),
      ])
def crop_rescale(ctx, clip, *, left, right, top, bottom, rescale):
    width, height = clip.width - left - right, clip.height - top - bottom
    if width <= 0 or height <= 0:
        raise ValueError(f"the crop removes the whole {clip.width}x{clip.height} frame")
    if left or right or top or bottom:
        # Odd crops of subsampled YUV are taken in 4:4:4.
        sub_w, sub_h = clip.format.subsampling_w, clip.format.subsampling_h
        if clip.format.color_family == vs.YUV and any(
                v % (1 << s) for v, s in ((left, sub_w), (right, sub_w), (top, sub_h), (bottom, sub_h))):
            clip = core.resize.Point(clip, format=clip.format.replace(subsampling_w=0, subsampling_h=0).id)
        clip = core.std.Crop(clip, left, right, top, bottom)
    if rescale != 1.0:
        out_w, out_h = max(1, round(width * rescale)), max(1, round(height * rescale))
        if clip.format.color_family == vs.YUV and (out_w % (1 << clip.format.subsampling_w)
                                                    or out_h % (1 << clip.format.subsampling_h)):
            clip = core.resize.Point(clip, format=clip.format.replace(subsampling_w=0, subsampling_h=0).id)
        clip = core.resize.Bicubic(clip, out_w, out_h, **CUBIC)
    return {"clip": clip}


def tag_frame_numbers(clip, prop="LRFrame"):
    """Stamps every frame with its own number, so it can be traced after frames are moved around."""
    def stamp(n, f):
        out = f.copy()
        out.props[prop] = n
        return out
    return core.std.ModifyFrame(clip, clip, stamp)


def remap(clip, length, frame_for):
    """
    A clip of `length` frames whose frame n is clip[frame_for(n)]. Built with
    FrameEval over shifted copies of the clip, one per distinct shift, so a
    long video needs no splice of single frames.
    """
    shifted = {}
    last = clip.num_frames - 1

    def shift(delta):
        if delta not in shifted:
            if delta > 0:
                shifted[delta] = clip[delta:] + clip[-1] * delta
            elif delta < 0:
                shifted[delta] = clip[0] * -delta + clip
            else:
                shifted[delta] = clip
        return shifted[delta]

    def select(n):
        m = min(max(frame_for(n), 0), last)
        return shift(m - n)

    base = core.std.BlankClip(clip, length=length)
    return core.std.FrameEval(base, select)


def section_offset(sections, t):
    offset = sections[0]["offset"]
    for section in sections:
        if t >= section["from"]:
            offset = section["offset"]
    return offset


def main_framing(framing, hr_duration):
    """The framing section covering most of the video."""
    if not framing:
        return {"left": 0.0, "top": 0.0, "width": 1.0, "height": 1.0}
    bounds = [f["from"] for f in framing] + [hr_duration]
    return max(framing, key=lambda f: bounds[framing.index(f) + 1] - f["from"])


def matching_views(hr, lr, framing, fmt):
    """
    HR and LR cut to the part of the picture both show, at one small size,
    for frame matching. `framing` says where HR's picture sits in LR's, as
    fractions of LR (align_videos.py's framing).
    """
    left, top, width, height = framing["left"], framing["top"], framing["width"], framing["height"]
    # The overlap in LR fractions, and the same region in HR fractions.
    x0, x1 = max(0.0, left), min(1.0, left + width)
    y0, y1 = max(0.0, top), min(1.0, top + height)
    if x1 - x0 < 0.2 or y1 - y0 < 0.2:
        x0, x1, y0, y1 = 0.0, 1.0, 0.0, 1.0
        left, top, width, height = 0.0, 0.0, 1.0, 1.0
    aspect = (y1 - y0) * lr.height / max(1e-6, (x1 - x0) * lr.width)
    out_w = MATCH_WIDTH
    out_h = max(16, round(out_w * aspect / 2) * 2)

    def view(clip, fx0, fx1, fy0, fy1):
        kwargs = {}
        if clip.format.color_family == vs.YUV and fmt == vs.RGBS:
            kwargs = {"matrix_in_s": "709"}
        elif clip.format.color_family == vs.RGB and fmt == vs.GRAYS:
            kwargs = {"matrix_s": "709"}
        return core.resize.Bicubic(clip, out_w, out_h, format=fmt,
                                   src_left=fx0 * clip.width, src_top=fy0 * clip.height,
                                   src_width=(fx1 - fx0) * clip.width, src_height=(fy1 - fy0) * clip.height,
                                   **kwargs)

    lr_view = view(lr, x0, x1, y0, y1)
    hr_view = view(hr, (x0 - left) / width, (x1 - left) / width, (y0 - top) / height, (y1 - top) / height)
    return hr_view, lr_view


@node("align_time", category="Align", label="Align LR to HR in Time",
      description="Finds which LR frame shows each HR frame, and returns the LR clip retimed to one frame per "
                  "HR frame. Each frame carries its source frame number as the LRFrame property.",
      inputs={"hr": "clip", "lr": "clip"}, outputs={"lr": "clip"},
      params=[
          Param("radius", "int", 3, help="Frames searched either side of the coarse timing for the exact match.",
                min=0),
          Param("precision", "choice", 1, choices=[1, 2, 3],
                help="vs_align's matching: 1 compares pixels (fast), 2 Butteraugli (needs the vship plugin "
                     "for CUDA or julek for the CPU), 3 TOPIQ (handles colour and warping differences; slowest; "
                     "its model is licensed CC BY-NC-SA 4.0, for non-commercial use only)."),
          Param("backend", "choice", "auto", choices=BACKENDS,
                help="'auto' uses CUDA when PyTorch sees a GPU, else the CPU."),
      ])
def align_time(ctx, hr, lr, *, radius, precision, backend):
    import vs_align
    backend = resolve_backend(backend)
    module = _align_videos()
    with _quiet_align(module, ctx, backend):
        try:
            coarse = module.align(hr, lr)
        except module.AlignError as e:
            raise ValueError(f"{ctx.video_name}: {e}") from e
    speed, sections = coarse["speed"], coarse["sections"]
    hr_fps, lr_fps = coarse["mainFps"], coarse["refFps"]
    for key in ("speed", "sections", "framing", "residualFrames"):
        ctx.note(key, coarse[key])
    ctx.message(f"{ctx.video_name}: LR = {speed:.5f} x HR {sections[0]['offset']:+.3f} s"
                + (f", {len(sections)} sections" if len(sections) > 1 else ""))

    def lr_frame(n):
        t = n / hr_fps
        return int(round((speed * t + section_offset(sections, t)) * lr_fps))

    tagged = tag_frame_numbers(lr)
    coarse_lr = remap(tagged, hr.num_frames, lr_frame)
    framing = main_framing(coarse["framing"], hr.num_frames / hr_fps)
    fmt = vs.GRAYS if int(precision) == 1 else vs.RGBS
    hr_view, lr_view = matching_views(hr, coarse_lr, framing, fmt)
    matched = vs_align.temporal(lr_view, hr_view, out=coarse_lr, tr=int(radius), precision=int(precision),
                                backend=backend)
    return {"lr": core.std.AssumeFPS(matched, hr)}


@node("align_space", category="Align", label="Align HR to LR in Space",
      description="Warps HR onto the LR picture upscaled by the training scale (XFeat and RIFE, from vs_align). "
                  "The output is the HR target at exactly scale x the LR size.",
      inputs={"hr": "clip", "lr": "clip"}, outputs={"hr": "clip"},
      params=[
          Param("scale", "int", 2, help="HR size as a multiple of LR.", min=1),
          Param("precision", "int", 3, help="vs_align's spatial precision, 1 to 4: higher is more exact and slower.",
                min=1, max=4),
          Param("wide_search", "bool", True,
                help="Match feature points first, so different crops, shears and rotations line up."),
          Param("lq_input", "bool", True,
                help="Follow general shapes rather than noise, grain or compression artefacts."),
          Param("backend", "choice", "auto", choices=BACKENDS,
                help="'auto' uses CUDA when PyTorch sees a GPU, else the CPU."),
      ])
def align_space(ctx, hr, lr, *, scale, precision, wide_search, lq_input, backend):
    import vs_align
    if hr.num_frames != lr.num_frames:
        raise ValueError(f"HR and LR differ in length ({hr.num_frames} and {lr.num_frames} frames); "
                         "align them in time first")
    backend = resolve_backend(backend)
    ref = core.resize.Bicubic(to_rgbs(lr), lr.width * int(scale), lr.height * int(scale), **CUBIC)
    aligned = vs_align.spatial(to_rgbs(hr), ref, precision=int(precision), wide_search=bool(wide_search),
                               lq_input=bool(lq_input), backend=backend)
    # Rounded rather than dithered: the target should carry no added noise.
    return {"hr": core.resize.Point(aligned, format=vs.RGB24, dither_type="none")}
