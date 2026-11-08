"""
Colour damage: how formats, decoders and analogue gear lose chroma detail.

chroma_subsample  a 4:2:0 (or 4:2:2, 4:1:1) round trip as a disc and its
                  player do it. A clip that is already in the chosen format
                  (a decoded codec output) is only upsampled again. The
                  upsample is where real players go wrong: point or bilinear
                  chroma, the wrong chroma siting, or the interlaced chroma
                  error (4:2:0 made per field, upsampled as if progressive,
                  which streaks colour edges). The result is 4:4:4, so the
                  damage survives until the dataset is written.
chroma_bleed      analogue colour bandwidth and Y/C delay: the chroma planes
                  blurred and shifted sideways against luma.
"""

import vapoursynth as vs

from ..framefx import gaussian_blur
from ..registry import Param, node
from .mixes import MIX, draw

core = vs.core

FORMATS = {"4:2:0": (1, 1), "4:2:2": (1, 0), "4:1:1": (2, 0)}
DOWN = {"bicubic": 0.5, "bilinear": 0.3, "box": 0.2}
UP = {"point": 0.25, "bilinear": 0.5, "bicubic": 0.25}
ZIMG = {"bicubic": "bicubic", "bilinear": "bilinear", "box": "point", "point": "point"}
# MPEG-2 and H.264 site 4:2:0 chroma left; the wrong guesses a decoder makes.
WRONG_SITING = ["center", "top_left"]


def need_yuv(clip, what):
    if clip.format.color_family != vs.YUV:
        raise ValueError(f"{what} works on YUV clips, got {clip.format.name}")


def with_subsampling(clip, ssw, ssh):
    fmt = clip.format
    return core.query_video_format(vs.YUV, fmt.sample_type, fmt.bits_per_sample, ssw, ssh).id


def subsample(clip, target, down):
    """4:4:4 to the target format. 'box' averages the chroma pixels it merges."""
    if down == "box":
        ssw, ssh = core.get_video_format(target).subsampling_w, core.get_video_format(target).subsampling_h
        clip = core.std.BoxBlur(clip, planes=[1, 2], hradius=(1 << ssw) // 2, vradius=(1 << ssh) // 2)
    return core.resize.Bicubic(clip, format=target, resample_filter_uv=ZIMG[down], chromaloc_s="left")


@node("chroma_subsample", category="Degrade", label="Chroma Subsampling", probability=True,
      inputs={"clip": "clip"}, outputs={"clip": "clip"},
      params=[
          Param("format", "choice", "4:2:0", choices=list(FORMATS), help="4:2:0 for DVD, broadcast and web."),
          Param("down", "choice", MIX, choices=[MIX, *DOWN], help="How the encoder shrank the chroma."),
          Param("up", "choice", MIX, choices=[MIX, *UP], help="How the player enlarges it again."),
          Param("wrong_siting", "float", 0.25, help="Chance the player places chroma half a pixel off.",
                min=0.0, max=1.0),
          Param("interlaced_error", "float", 0.0,
                help="Chance of the interlaced chroma error (colour streaks on moving edges).", min=0.0, max=1.0),
      ])
def chroma_subsample(ctx, clip, *, format, down, up, wrong_siting, interlaced_error):
    need_yuv(clip, "chroma subsampling")
    if format not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}")
    rng = ctx.rng
    ssw, ssh = FORMATS[format]
    target = with_subsampling(clip, ssw, ssh)
    full = with_subsampling(clip, 0, 0)
    up_kernel = draw(rng, up, UP)
    siting = rng.choice(WRONG_SITING) if rng.random() < wrong_siting else "left"
    interlaced = ssh > 0 and rng.random() < interlaced_error
    ctx.note("up", up_kernel)
    ctx.note("siting", siting)
    ctx.note("interlaced_error", interlaced)

    if clip.format.id != target or interlaced:
        down_kernel = draw(rng, down, DOWN)
        ctx.note("down", down_kernel)
        source = core.resize.Bicubic(clip, format=full) if clip.format.id != full else clip
        if interlaced:
            fields = core.std.SeparateFields(source, tff=True)
            clip = core.std.DoubleWeave(subsample(fields, target, down_kernel), tff=True)[::2]
        else:
            clip = subsample(source, target, down_kernel)
    clip = core.resize.Bicubic(clip, format=full, resample_filter_uv=ZIMG[up_kernel], chromaloc_in_s=siting)
    return {"clip": core.std.RemoveFrameProps(clip, ["_Field", "_FieldBased"])}


@node("chroma_bleed", category="Degrade", label="Chroma Bleed", probability=True,
      inputs={"clip": "clip"}, outputs={"clip": "clip"},
      params=[
          Param("blur", "float", {"uniform": [0.5, 1.5]}, random=True,
                help="Sideways chroma blur, sigma in luma pixels.", min=0.0, max=8.0),
          Param("shift", "float", {"uniform": [-1.0, 1.0]}, random=True,
                help="Sideways chroma shift in luma pixels; positive moves colour right.", min=-8.0, max=8.0),
      ])
def chroma_bleed(ctx, clip, *, blur, shift):
    need_yuv(clip, "chroma bleed")
    ctx.note("blur", round(blur, 3))
    ctx.note("shift", round(shift, 3))
    factor = 1 << clip.format.subsampling_w
    clip = gaussian_blur(clip, blur / factor, 0.0, planes=[1, 2])
    if abs(shift) < 0.01:
        return {"clip": clip}
    chroma = [core.std.ShufflePlanes(clip, plane, vs.GRAY) for plane in (1, 2)]
    moved = [core.resize.Bicubic(p, src_left=-shift / factor, filter_param_a=0.0, filter_param_b=0.5)
             for p in chroma]
    return {"clip": core.std.ShufflePlanes([clip, *moved], [0, 0, 0], vs.YUV)}
