"""
Helpers for degradations that work on whole frames: shifting a clip in time,
picking frames by number, rewriting planes with NumPy where no VapourSynth filter does the job, and
Gaussian blurs of any width.
"""

import math

import numpy as np
import vapoursynth as vs

core = vs.core


def shifted(clip, offset):
    """
    A clip whose frame n is frame n + offset of `clip`; the first or last
    frame is repeated where the shift runs off the ends.
    """
    offset = max(-(clip.num_frames - 1), min(clip.num_frames - 1, int(offset)))
    if offset < 0:
        return clip[0] * -offset + clip[:offset]
    if offset > 0:
        return clip[offset:] + clip[-1] * offset
    return clip


def picked(clip, indices):
    """
    A clip of len(indices) frames whose frame n is frame indices[n] of
    `clip`. It selects from shifted copies of the clip, one per distinct
    shift, each made when it is first needed.
    """
    copies = {}

    def select(n):
        delta = indices[n] - n
        if delta not in copies:
            copies[delta] = clip[0] * -delta + clip if delta < 0 else clip[delta:]
        return copies[delta]
    return core.std.FrameEval(core.std.BlankClip(clip, length=len(indices)), select)


def modify_planes(clip, planes, func, others=()):
    """
    Rewrites the given planes of every frame. func(n, plane, arrays) gets the
    plane of `clip` and of each clip in `others` for frame n, as 2-D arrays,
    and returns the new plane; the other planes are copied as they are.
    """
    def select(n, f):
        frames = f if isinstance(f, list) else [f]
        out = frames[0].copy()
        for plane in planes:
            np.asarray(out[plane])[:] = func(n, plane, [np.asarray(fr[plane]) for fr in frames])
        return out
    return core.std.ModifyFrame(clip, clips=[clip, *others], selector=select)


def split_planes(clip):
    return [core.std.ShufflePlanes(clip, i, vs.GRAY) for i in range(clip.format.num_planes)]


def merge_rgb(planes):
    return core.std.ShufflePlanes(planes, [0, 0, 0], vs.RGB)


def to_rgbs(clip):
    """The clip as float RGB, full range; a YUV clip is read as BT.709."""
    if clip.format.id == vs.RGBS:
        return clip
    kwargs = {"matrix_in_s": "709"} if clip.format.color_family == vs.YUV else {}
    return core.resize.Bicubic(clip, format=vs.RGBS, **kwargs)


def from_rgbs(rgbs, like):
    """Float RGB converted to the format of `like`, dithered."""
    if like.format.id == rgbs.format.id:
        return rgbs
    kwargs = {"matrix_s": "709"} if like.format.color_family == vs.YUV else {}
    return core.resize.Bicubic(rgbs, format=like.format.id, dither_type="error_diffusion", **kwargs)


# std.Convolution takes up to 25 taps; a wider Gaussian is several narrower
# passes, whose variances add up to the wide one's.
MAX_PASS_SIGMA = 3.0


def gaussian_taps(sigma):
    radius = max(1, math.ceil(sigma * 4))
    return [math.exp(-(i * i) / (2 * sigma * sigma)) for i in range(-radius, radius + 1)]


def gaussian_blur(clip, sigma_h, sigma_v=None, planes=None):
    """
    A Gaussian blur with its own sigma across and down, in pixels of each
    plane; a sigma of 0 leaves that direction alone.
    """
    sigma_v = sigma_h if sigma_v is None else sigma_v
    planes = list(range(clip.format.num_planes)) if planes is None else list(planes)
    for sigma, mode in ((sigma_h, "h"), (sigma_v, "v")):
        if sigma <= 0.05:
            continue
        split = math.ceil((sigma / MAX_PASS_SIGMA) ** 2)
        taps = gaussian_taps(sigma / math.sqrt(split))
        for _ in range(split):
            clip = core.std.Convolution(clip, matrix=taps, mode=mode, planes=planes)
    return clip
