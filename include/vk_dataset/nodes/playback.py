"""
What a software player or a TV does to decoded video before showing it:
deblocking, which smooths across the 8x8 block edges and leaves soft,
waxy flat areas, and noise reduction, which smears fine detail and leaves
faint trails on motion.

player_deblock  the MPEG-2 / H.264 style in-loop deblock (vapoursynth-deblock),
                its quantizer setting from the strength.
player_dnr      FluxSmooth (zsmooth): each pixel averaged with its neighbours
                in space and time that are within a threshold of it.

A player that does both is a Player Deblock followed by a Player DNR.
"""

import vapoursynth as vs

from ..registry import Param, node
from .mixes import as_integer_yuv

core = vs.core


def strength_param():
    return Param("strength", "float", {"uniform": [0.2, 0.8]}, random=True, help="0 is light, 1 heavy.",
                 min=0.0, max=1.0)


@node("player_deblock", category="Degrade", label="Player Deblock", probability=True,
      inputs={"clip": "clip"}, outputs={"clip": "clip"}, params=[strength_param()])
def player_deblock(ctx, clip, *, strength):
    clip, restore = as_integer_yuv(clip)
    quant = round(20 + 30 * strength)
    ctx.note("strength", round(strength, 3))
    ctx.note("deblock_quant", quant)
    return {"clip": restore(core.deblock.Deblock(clip, quant=quant))}


@node("player_dnr", category="Degrade", label="Player DNR", probability=True,
      inputs={"clip": "clip"}, outputs={"clip": "clip"}, params=[strength_param()])
def player_dnr(ctx, clip, *, strength):
    clip, restore = as_integer_yuv(clip)
    temporal, spatial = round(3 + 9 * strength, 1), round(2 + 6 * strength, 1)
    ctx.note("strength", round(strength, 3))
    ctx.note("dnr_thresholds", [temporal, spatial])
    return {"clip": restore(core.zsmooth.FluxSmoothST(clip, temporal_threshold=temporal, spatial_threshold=spatial))}
