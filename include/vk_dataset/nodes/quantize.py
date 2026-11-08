"""
Banding: the picture rounded to fewer levels than its format holds, as an
8-bit master pushed through a low-precision grade or encoder leaves smooth
gradients. Dithering trades the bands for fine noise: ordered (a fixed
pattern) or error diffusion.

zimg does the rounding and the dither: the values are divided by the step
between kept levels, so a full-range float to 8-bit conversion rounds to
whole steps (float chroma is centred on 0, hence the 128), and multiplied
back afterwards.
"""

import vapoursynth as vs

from ..registry import Param, node
from .mixes import MIX, as_integer_yuv, draw

core = vs.core

DITHERS = {"none": 0.5, "ordered": 0.2, "error diffusion": 0.3}
ZIMG_DITHER = {"none": "none", "ordered": "ordered", "error diffusion": "error_diffusion"}


@node("quantize", category="Degrade", label="Banding", probability=True,
      inputs={"clip": "clip"}, outputs={"clip": "clip"},
      params=[
          Param("bits", "int", {"int": [6, 7]}, random=True, help="Levels kept, in bits; 8 changes nothing.",
                min=3, max=8),
          Param("dither", "choice", MIX, choices=[MIX, *DITHERS], help="How the rounding is hidden, if at all."),
      ])
def quantize(ctx, clip, *, bits, dither):
    source = clip
    clip, restore = as_integer_yuv(clip)
    fmt = clip.format
    bits = max(1, min(int(bits), 8))
    chosen = draw(ctx.rng, dither, DITHERS)
    ctx.note("bits", bits)
    ctx.note("dither", chosen)
    if bits >= 8 and fmt.bits_per_sample == 8:
        return {"clip": source}
    step = (1 << fmt.bits_per_sample) / (1 << bits)
    floats = core.std.Expr(clip, [f"x {step} / 255 /", f"x {step} / 128 - 255 /"],
                           format=core.query_video_format(vs.YUV, vs.FLOAT, 32, fmt.subsampling_w,
                                                          fmt.subsampling_h).id)
    levels = core.resize.Point(floats, format=core.query_video_format(vs.YUV, vs.INTEGER, 8, fmt.subsampling_w,
                                                                      fmt.subsampling_h).id,
                               range_in_s="full", range_s="full", dither_type=ZIMG_DITHER[chosen])
    # Integer output is clamped to the format's range by Expr itself. The
    # props go back to the input's, since the rounding step tagged full range.
    return {"clip": restore(core.std.CopyFrameProps(core.std.Expr(levels, f"x {step} *", format=fmt.id), clip))}
