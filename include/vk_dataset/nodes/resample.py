"""
Scaling the LR side the way mastering, players and encoders scale video.

resample        to a DVD frame (720x480 or 720x576, anamorphic: the picture
                is squeezed or stretched to fill it, as on a disc) or by a
                fraction of the input.
scale_to_model  to the size the model takes as input: the HR size divided by
                the model's scale. It is the last scaling of an LR chain.

Each draws its kernel once per chunk. "mix" picks from the kernels real
scalers use, with the kernel's own settings varied (bicubic b and c,
lanczos taps), so the model does not learn to invert one exact filter.
Softness widens the kernel by a Gaussian; light chooses whether scaling
happens on the coded values (what most software does) or in linear light.
"""

import math

import vapoursynth as vs

from ..framefx import gaussian_blur
from ..registry import Param, node
from .mixes import LIGHTS, MIX, draw, draw_light, from_working, subsampled_size, to_working

core = vs.core

KERNELS = ["bicubic", "bilinear", "lanczos", "spline36", "area", "point"]
# Mastering and player scalers. Point is left out: a model trained on it
# learns to undo stair-stepping that real video rarely has.
KERNEL_MIX = {"bicubic": 0.35, "bilinear": 0.2, "lanczos": 0.2, "spline36": 0.15, "area": 0.1}
UPSCALE_MIX = {"bicubic": 0.35, "bilinear": 0.25, "lanczos": 0.2, "spline36": 0.2}

DVD_SIZES = {"720x480": (720, 480), "720x576": (720, 576)}
SIZES = ["720x480", "720x576", "720x480 or 720x576", "scale"]


def kernel_settings(rng, name):
    """The zimg filter and arguments for a kernel, with its shape drawn for this chunk."""
    if name == "bicubic":
        b, c = rng.uniform(0.0, 0.33), rng.uniform(0.33, 0.6)
        return "Bicubic", {"filter_param_a": round(b, 3), "filter_param_b": round(c, 3)}
    if name == "lanczos":
        return "Lanczos", {"filter_param_a": rng.randint(2, 4)}
    return {"bilinear": "Bilinear", "spline36": "Spline36", "point": "Point", "area": "Bilinear"}[name], {}


def scale(clip, width, height, name, settings, softness):
    """
    Scales with a zimg kernel. 'area' averages whole pixels when shrinking,
    as box scalers do: a box blur as wide as the ratio before a bilinear
    resize. Softness above 1 adds a Gaussian on the low-resolution side.
    """
    filter_name, kwargs = settings
    ratio_x, ratio_y = clip.width / width, clip.height / height
    extra = 0.5 * math.sqrt(max(0.0, softness * softness - 1.0))
    if ratio_x > 1 or ratio_y > 1:
        clip = gaussian_blur(clip, extra * max(1.0, ratio_x), extra * max(1.0, ratio_y), planes=_planes(clip))
        if name == "area":
            rx, ry = max(0, round(ratio_x / 2 - 0.5)), max(0, round(ratio_y / 2 - 0.5))
            if rx or ry:
                clip = core.std.BoxBlur(clip, hradius=rx, vradius=ry)
    fmt = subsampled_size(clip, width, height)
    clip = getattr(core.resize, filter_name)(clip, width, height, format=fmt, **kwargs)
    if ratio_x <= 1 and ratio_y <= 1:
        clip = gaussian_blur(clip, extra, extra, planes=_planes(clip))
    return clip


def _planes(clip):
    return list(range(clip.format.num_planes))


def scale_in_light(ctx, clip, width, height, kernel, softness, light, mix):
    """Draws the kernel and light for this chunk, notes them, and scales."""
    name = draw(ctx.rng, kernel, {k: mix.get(k, 0.0) for k in KERNELS})
    settings = kernel_settings(ctx.rng, name)
    chosen = draw_light(ctx.rng, light)
    ctx.note("kernel", name)
    if settings[1]:
        ctx.note("kernel_settings", settings[1])
    ctx.note("light", chosen)
    ctx.note("size", [width, height])
    if (width, height) == (clip.width, clip.height):
        return clip
    if chosen == "gamma" and clip.format.color_family != vs.RGB:
        return scale(clip, width, height, name, settings, softness)
    working = scale(to_working(clip, chosen), width, height, name, settings, softness)
    return from_working(working, core.std.BlankClip(clip, format=subsampled_size(clip, width, height)), chosen)


def even(value):
    return max(2, int(round(value)) // 2 * 2)


KERNEL_PARAM_HELP = "'mix' picks a realistic kernel per chunk."
LIGHT_HELP = "Scale coded values (gamma, like most software) or linear light."


@node("resample", category="Degrade", label="Resample", probability=True,
      inputs={"clip": "clip"}, outputs={"clip": "clip"},
      params=[
          Param("size", "choice", "720x480 or 720x576", choices=SIZES,
                help="A DVD frame (anamorphic), or 'scale' for a fraction of the input."),
          Param("scale", "float", 0.5, random=True, help="For 'scale': output size as a fraction of the input.",
                min=0.05, max=4.0),
          Param("kernel", "choice", MIX, choices=[MIX, *KERNELS], help=KERNEL_PARAM_HELP),
          Param("softness", "float", {"uniform": [1.0, 1.15]}, random=True,
                help="1 is the kernel as it is; higher blurs it wider.", min=1.0, max=2.0),
          Param("light", "choice", "mostly gamma", choices=LIGHTS, help=LIGHT_HELP),
      ])
def resample(ctx, clip, *, size, scale, kernel, softness, light):
    if size == "scale":
        width, height = even(clip.width * scale), even(clip.height * scale)
    else:
        if size not in SIZES:
            raise ValueError(f"size must be one of {', '.join(SIZES)}")
        key = size if size in DVD_SIZES else ctx.rng.choice(list(DVD_SIZES))
        width, height = DVD_SIZES[key]
    ctx.note("softness", round(softness, 3))
    return {"clip": scale_in_light(ctx, clip, width, height, kernel, softness, light, KERNEL_MIX)}


@node("scale_to_model", category="Degrade", label="Scale to Model Size",
      inputs={"clip": "clip", "hr": "clip"}, outputs={"clip": "clip"},
      params=[
          Param("scale", "choice", 2, choices=[1, 2, 3, 4], help="The model's upscale factor: LR is HR / scale."),
          Param("kernel", "choice", MIX, choices=[MIX, *KERNELS], help=KERNEL_PARAM_HELP),
          Param("softness", "float", {"uniform": [1.0, 1.1]}, random=True,
                help="1 is the kernel as it is; higher blurs it wider.", min=1.0, max=2.0),
          Param("light", "choice", "mostly gamma", choices=LIGHTS, help=LIGHT_HELP),
      ])
def scale_to_model(ctx, clip, hr, *, scale, kernel, softness, light):
    factor = int(scale)
    if factor < 1:
        raise ValueError("the model scale must be 1 or more")
    width, height = hr.width // factor, hr.height // factor
    mix = KERNEL_MIX if width < clip.width or height < clip.height else UPSCALE_MIX
    ctx.note("softness", round(softness, 3))
    return {"clip": scale_in_light(ctx, clip, width, height, kernel, softness, light, mix)}
