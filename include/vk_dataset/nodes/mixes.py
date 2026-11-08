"""
Shared pieces of the degradation nodes: drawing one option from a weighted
mix, and moving a clip into the light (gamma or linear) a filter works in.

A choice setting of "mix" draws from the node's own realistic weights once
per chunk; any other value is used as it is.
"""

import vapoursynth as vs

core = vs.core

MIX = "mix"

LIGHTS = ["gamma", "linear", "mostly gamma", "mostly linear"]


def draw(rng, value, weights):
    """`value`, or for "mix" one key of `weights` drawn by weight."""
    if value != MIX:
        if value not in weights:
            raise ValueError(f"unknown option {value!r}; known: mix, {', '.join(weights)}")
        return value
    names = [name for name, weight in weights.items() if weight > 0]
    return rng.choices(names, weights=[weights[name] for name in names])[0]


def draw_light(rng, light):
    """'gamma' or 'linear'; the "mostly" settings pick their light 80% of the time."""
    if light in ("gamma", "linear"):
        return light
    if light not in LIGHTS:
        raise ValueError(f"light must be one of {', '.join(LIGHTS)}")
    usual = light.split()[1]
    other = "linear" if usual == "gamma" else "gamma"
    return usual if rng.random() < 0.8 else other


def to_working(clip, light):
    """The clip as float RGB, in linear light or as coded (BT.709)."""
    kwargs = {"matrix_in_s": "709"} if clip.format.color_family == vs.YUV else {}
    if light == "linear":
        kwargs.update(transfer_in_s="709", transfer_s="linear")
    return core.resize.Bicubic(clip, format=vs.RGBS, **kwargs)


def from_working(rgbs, like, light):
    """Float RGB from to_working back to the format of `like`, dithered, at the float clip's size."""
    kwargs = {"matrix_s": "709"} if like.format.color_family == vs.YUV else {}
    if light == "linear":
        kwargs.update(transfer_in_s="linear", transfer_s="709")
    return core.resize.Bicubic(rgbs, format=like.format.id, dither_type="error_diffusion", **kwargs)


def as_integer_yuv(clip):
    """
    The clip as integer YUV, and a function that turns a result back into
    the clip's own format. RGB and float clips go through 16-bit 4:4:4.
    """
    fmt = clip.format
    if fmt.color_family == vs.YUV and fmt.sample_type == vs.INTEGER:
        return clip, lambda result: result
    kwargs = {"matrix_s": "709"} if fmt.color_family == vs.RGB else {}
    back = {"matrix_in_s": "709"} if fmt.color_family == vs.RGB else {}
    yuv = core.resize.Bicubic(clip, format=vs.YUV444P16, **kwargs)
    return yuv, lambda result: core.resize.Bicubic(result, format=fmt.id, **back)


def subsampled_size(clip, width, height):
    """
    The clip's format if it can hold width x height, else its 4:4:4 version:
    an odd size cannot be subsampled.
    """
    fmt = clip.format
    if width % (1 << fmt.subsampling_w) or height % (1 << fmt.subsampling_h):
        return core.query_video_format(fmt.color_family, fmt.sample_type, fmt.bits_per_sample, 0, 0).id
    return fmt.id
