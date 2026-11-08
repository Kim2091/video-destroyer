"""
Multipath ghosting on an analogue signal: a reflection of the picture
arrives late, so a faint, soft copy sits a fixed distance to the right of
every edge. A negative strength is an inverted echo, which the receiver
also shows. The echo replaces part of the picture rather than adding to
it, so the brightness stays the same.
"""

import vapoursynth as vs

from ..framefx import gaussian_blur
from ..registry import Param, node

core = vs.core

POLARITIES = ["positive", "negative"]


@node("ghost_echo", category="Degrade", label="Ghost Echo", probability=True,
      inputs={"clip": "clip"}, outputs={"clip": "clip"},
      params=[
          Param("strength", "float", {"uniform": [0.03, 0.08]}, random=True, help="Echo strength, 0 to 1.",
                min=0.0, max=0.5),
          Param("delay", "float", {"uniform": [8.0, 40.0]}, random=True,
                help="How far right the echo sits, in pixels.", min=1.0, max=200.0),
          Param("blur", "float", {"uniform": [1.0, 3.0]}, random=True, help="Echo softness, sigma in pixels.",
                min=0.0, max=8.0),
          Param("polarity", "choice", {"choice": POLARITIES, "weights": [3, 2]}, choices=POLARITIES,
                help="Positive, or inverted as on some sets."),
      ])
def ghost_echo(ctx, clip, *, strength, delay, blur, polarity):
    if clip.format.color_family != vs.YUV:
        raise ValueError(f"ghost echo works on YUV clips, got {clip.format.name}")
    if polarity not in POLARITIES:
        raise ValueError(f"polarity must be one of {', '.join(POLARITIES)}")
    gain = -strength if polarity == "negative" else strength
    ctx.note("strength", round(gain, 4))
    ctx.note("delay", round(delay, 2))
    ctx.note("blur", round(blur, 3))
    planes = []
    for index in range(clip.format.num_planes):
        plane = core.std.ShufflePlanes(clip, index, vs.GRAY)
        factor = 1 << (clip.format.subsampling_w if index else 0)
        echo = core.resize.Bicubic(gaussian_blur(plane, blur / factor, blur / factor), src_left=-delay / factor)
        planes.append(core.std.Expr([plane, echo], f"x y x - {gain} * +"))
    return {"clip": core.std.ShufflePlanes(planes, [0, 0, 0], vs.YUV)}
