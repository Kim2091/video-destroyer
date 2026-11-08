"""
Edge enhancement as DVD mastering, cameras and broadcast gear apply it: a
Gaussian unsharp mask on luma, out = in + amount * (in - blur(in)), which
leaves bright and dark halos beside edges. Horizontal-only sharpening is the
analogue and broadcast kind (a peaking filter along the scan line).
Differences below the threshold are left alone, as coring does.
"""

import vapoursynth as vs

from ..framefx import gaussian_blur
from ..registry import Param, node
from .mixes import MIX, draw

core = vs.core

DIRECTIONS = {"2d": 0.7, "horizontal": 0.3}


@node("sharpen", category="Degrade", label="Sharpen (Halos)", probability=True,
      inputs={"clip": "clip"}, outputs={"clip": "clip"},
      params=[
          Param("radius", "float", {"uniform": [0.7, 1.8]}, random=True,
                help="Gaussian sigma in pixels; wider makes wider halos.", min=0.3, max=6.0),
          Param("amount", "float", {"uniform": [0.4, 1.2]}, random=True, help="Sharpening strength.",
                min=0.0, max=4.0),
          Param("threshold", "float", {"uniform": [0.0, 2.0]}, random=True,
                help="Edges weaker than this (8-bit levels) are not sharpened.", min=0.0, max=20.0),
          Param("direction", "choice", MIX, choices=[MIX, *DIRECTIONS],
                help="2D, or horizontal only like analogue peaking."),
      ])
def sharpen(ctx, clip, *, radius, amount, threshold, direction):
    if clip.format.color_family == vs.RGB:
        raise ValueError("sharpen works on YUV or gray clips")
    chosen = draw(ctx.rng, direction, DIRECTIONS)
    for key, value in (("radius", radius), ("amount", amount), ("threshold", threshold)):
        ctx.note(key, round(value, 3))
    ctx.note("direction", chosen)
    blurred = gaussian_blur(clip, radius, radius if chosen == "2d" else 0.0, planes=[0])
    fmt = clip.format
    peak = (1 << fmt.bits_per_sample) - 1 if fmt.sample_type == vs.INTEGER else 1.0
    level = threshold * peak / 255.0
    # Integer output is clamped to the format's range by Expr itself.
    expression = f"x y - abs {level} < x x x y - {amount} * + ?"
    return {"clip": core.std.Expr([clip, blurred], [expression] + [""] * (fmt.num_planes - 1))}
