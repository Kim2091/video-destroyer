"""
Film grain and camera sensor noise. The statistics (strength, grain size,
colour share) are drawn once per chunk; the pattern itself is new on every
frame, as real grain and noise are, and is seeded from the chunk and the
frame number so a rerun gives the same frames.

Grain size comes from drawing the noise on a coarser grid and enlarging it
smoothly, which gives the clumps real grain and demosaiced noise have;
size 1 is independent per pixel. Chroma noise is drawn on the chroma
planes, so in 4:2:0 its clumps are twice as wide as luma's.

film_grain    strongest in the mid-tones, weaker in shadows and highlights,
              as grain shows through a print.
sensor_noise  shot noise (which grows with the light) plus read noise (which
              does not), moved from linear light into coded values, so
              shadows are noisier than highlights.
"""

import math

import cv2
import numpy as np
import vapoursynth as vs

from ..framefx import modify_planes
from ..registry import Param, node
from .mixes import as_integer_yuv

core = vs.core


def noise_field(seed, n, plane, shape, size):
    """A unit-variance Gaussian field for one plane of frame n, with clumps `size` pixels wide."""
    rng = np.random.default_rng([seed, n, plane])
    height, width = shape
    if size <= 1.05:
        return rng.standard_normal(shape, dtype=np.float32)
    small = rng.standard_normal((max(2, math.ceil(height / size)), max(2, math.ceil(width / size))),
                                dtype=np.float32)
    field = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    return field / max(float(field.std()), 1e-6)


def luma_levels(clip):
    """Black, white and the 8-bit level size of the luma plane, for limited-range integer YUV."""
    bits = clip.format.bits_per_sample
    scale = 1 << (bits - 8)
    return 16 * scale, 235 * scale, scale


def add_noise(ctx, clip, luma_sigma, chroma_sigma, size):
    """
    Adds noise to every plane. luma_sigma(v) gives the luma deviation in
    8-bit levels for luma v in 0..1; chroma_sigma is a constant in levels.
    """
    clip, restore = as_integer_yuv(clip)
    fmt = clip.format
    seed = ctx.rng.getrandbits(63)
    black, white, scale = luma_levels(clip)
    peak = (1 << fmt.bits_per_sample) - 1

    def func(n, plane, arrays):
        source = arrays[0]
        field = noise_field(seed, n, plane, source.shape, size)
        values = source.astype(np.float32)
        if plane == 0:
            v = np.clip((values - black) / (white - black), 0.0, 1.0)
            values += field * luma_sigma(v) * scale
        else:
            if chroma_sigma <= 0:
                return source
            values += field * chroma_sigma * scale
        return np.clip(np.rint(values), 0, peak).astype(source.dtype)
    return restore(modify_planes(clip, range(fmt.num_planes), func))


@node("film_grain", category="Degrade", label="Film Grain", probability=True,
      inputs={"clip": "clip"}, outputs={"clip": "clip"},
      params=[
          Param("strength", "float", {"uniform": [1.0, 3.0]}, random=True,
                help="Grain in mid-grey, in 8-bit levels (standard deviation).", min=0.0, max=20.0),
          Param("size", "float", {"uniform": [1.2, 2.2]}, random=True, help="Grain clump size in pixels.",
                min=1.0, max=6.0),
          Param("chroma", "float", {"uniform": [0.0, 0.3]}, random=True,
                help="Colour grain as a share of the brightness grain.", min=0.0, max=1.0),
      ])
def film_grain(ctx, clip, *, strength, size, chroma):
    for key, value in (("strength", strength), ("size", size), ("chroma", chroma)):
        ctx.note(key, round(value, 3))

    def luma_sigma(v):
        return strength * np.maximum(0.3, 4.0 * v * (1.0 - v))
    return {"clip": add_noise(ctx, clip, luma_sigma, strength * chroma, size)}


# Coded value v stands for light v ** GAMMA.
GAMMA = 2.4
# Below this coded value cameras switch to a linear toe, which keeps the
# shadow noise from growing without limit.
TOE = 0.08


@node("sensor_noise", category="Degrade", label="Sensor Noise", probability=True,
      inputs={"clip": "clip"}, outputs={"clip": "clip"},
      params=[
          Param("strength", "float", {"uniform": [0.5, 2.5]}, random=True,
                help="Noise in mid-grey, in 8-bit levels (standard deviation).", min=0.0, max=20.0),
          Param("shot_share", "float", {"uniform": [0.5, 0.9]}, random=True,
                help="Share of the mid-grey noise that grows with light; the rest is read noise.",
                min=0.0, max=1.0),
          Param("size", "float", {"uniform": [1.0, 1.5]}, random=True,
                help="Clump size in pixels; demosaicing makes it a little over 1.", min=1.0, max=4.0),
          Param("chroma", "float", {"uniform": [0.3, 1.0]}, random=True,
                help="Colour noise as a share of the brightness noise.", min=0.0, max=2.0),
      ])
def sensor_noise(ctx, clip, *, strength, shot_share, size, chroma):
    for key, value in (("strength", strength), ("shot_share", shot_share), ("size", size), ("chroma", chroma)):
        ctx.note(key, round(value, 3))
    mid_light = 0.5 ** GAMMA

    def luma_sigma(v):
        v = np.maximum(v, TOE)
        light = v ** GAMMA
        deviation = np.sqrt(shot_share * light / mid_light + (1.0 - shot_share))
        # Light to coded value: dv/dlight relative to mid-grey.
        slope = (v / 0.5) ** (1.0 - GAMMA)
        return strength * np.minimum(3.0, deviation * slope)
    return {"clip": add_noise(ctx, clip, luma_sigma, strength * chroma, size)}
