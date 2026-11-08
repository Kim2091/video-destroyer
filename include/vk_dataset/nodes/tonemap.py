"""
HDR to SDR tonemapping, ported from video-destroyer's tonemap. It is meant
for the HR side, before the chunk node, so that HR and LR are both SDR.

The original runs zscale to linear light in BT.709 primaries, ffmpeg's
tonemap filter, and zscale back to BT.709 in 8-bit 4:2:0. This node does the
same with zimg for the conversions and the tonemap filter's own formulas for
the curves, written as float expressions:

  1. desat: each colour is mixed towards its luma (the sum r + g + b, which is
     what ffmpeg 8 computes) by the amount
     max(luma - desat, 1e-6) / luma, so bright colours lose saturation.
  2. The brightest channel is mapped through the curve, and all three
     channels are scaled by the same factor.

Linear light is scaled so that 1.0 is 100 nits, as zscale does, and the
original hands target_nits to the filter as its `peak`, in those units. The
default of 100 therefore means a signal peak of 10,000 nits, the full PQ
range; this node keeps that meaning so a config gives the same picture.

ffmpeg's tonemap has no bt2390 curve, so the original fails on it. Here it
is the BT.2408 reference tone mapper, working in the PQ domain from a source
peak of target_nits * 100 nits to 100 nits.

With auto_detect on, a clip that is not HDR (no PQ or HLG transfer, no
BT.2020 matrix or primaries in its frame properties) passes through
unchanged. Untagged clips are read as BT.2020 PQ when auto_detect is off.
"""

import math

import vapoursynth as vs

from ..framefx import merge_rgb, split_planes
from ..registry import Param, node
from ..runner import NodeSkipped

core = vs.core

ALGORITHMS = ["hable", "mobius", "reinhard", "bt2390", "linear"]
LUMA = (1.0, 1.0, 1.0)
PQ_TRANSFER, HLG_TRANSFER = 16, 18
MOBIUS_KNEE = 0.3

PQ_M1, PQ_M2 = 2610 / 16384, 2523 / 4096 * 128
PQ_C1, PQ_C2, PQ_C3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32


def pq_encode(nits):
    """PQ signal of a luminance in nits (a number)."""
    y = (nits / 10000.0) ** PQ_M1
    return ((PQ_C1 + PQ_C2 * y) / (1 + PQ_C3 * y)) ** PQ_M2


def curve_stages(algorithm, peak):
    """
    The curve as expressions on x, the signal relative to 100 nits, to be
    applied one after the other.
    """
    if algorithm == "hable":
        top = (peak * (peak * 0.15 + 0.05) + 0.004) / (peak * (peak * 0.15 + 0.5) + 0.06) - 0.02 / 0.3
        return [f"x x 0.15 * 0.05 + * 0.004 + x x 0.15 * 0.5 + * 0.06 + / {0.02 / 0.3} - {top} /"]
    if algorithm == "reinhard":
        return [f"x x 1 + / {(peak + 1) / peak} *"]
    if algorithm == "linear":
        return [f"x {1.0 / peak} *"]
    if algorithm == "mobius":
        j = MOBIUS_KNEE
        a = -j * j * (peak - 1) / (j * j - 2 * j + peak)
        b = (j * j - 2 * j * peak + peak) / max(peak - 1, 1e-6)
        k = (b * b + 2 * b * j + j * j) / (b - a)
        return [f"x {j} <= x {k} x {a} + * x {b} + / ?"]
    return bt2390_stages(peak)


def bt2390_stages(peak):
    """BT.2408's EETF from a peak of `peak` * 100 nits down to 100 nits, in the PQ domain."""
    source = pq_encode(max(peak * 100.0, 100.0))
    target = pq_encode(100.0) / source
    knee = 1.5 * target - 0.5
    t = f"x {knee} - {1.0 / (1 - knee)} *"
    spline = (f"{2 * knee + (1 - knee) - 2 * target} {t} * {-3 * knee - 2 * (1 - knee) + 3 * target} + {t} * "
              f"{1 - knee} + {t} * {knee} +")
    return [
        f"x 100 * 10000 / {PQ_M1} pow",
        f"{PQ_C1} {PQ_C2} x * + 1 {PQ_C3} x * + / {PQ_M2} pow {1.0 / source} *",
        f"x {knee} < x {spline} ?",
        f"x {source} * {1.0 / PQ_M2} pow dup {PQ_C1} - 0 max swap {PQ_C3} * {PQ_C2} swap - / "
        f"{1.0 / PQ_M1} pow 10000 * 100 /",
    ]


def tonemap_planes(r, g, b, algorithm, desat, peak):
    """
    The tonemapped planes, from the three linear RGB planes (BT.709
    primaries, 1.0 = 100 nits), as ffmpeg's tonemap filter computes them.
    """
    luma = core.std.Expr([r, g, b], f"x {LUMA[0]} * y {LUMA[1]} * + z {LUMA[2]} * +")
    overbright = core.std.Expr(luma, f"x {desat} - 1e-6 max x 1e-6 max /")
    r, g, b = (core.std.Expr([c, luma, overbright], "x 1 z - * y z * +") for c in (r, g, b))
    signal = core.std.Expr([r, g, b], "x y max z max 0 max")
    mapped = signal
    for stage in curve_stages(algorithm, peak):
        mapped = core.std.Expr(mapped, stage)
    scale = core.std.Expr([mapped, signal], "x y 1e-6 max /")
    return [core.std.Expr([c, scale], "x y *") for c in (r, g, b)]


def source_description(frame):
    props = frame.props
    transfer = props.get("_Transfer", 2)
    matrix = props.get("_Matrix", 2)
    primaries = props.get("_Primaries", 2)
    is_hdr = transfer in (PQ_TRANSFER, HLG_TRANSFER) or matrix in (9, 10) or primaries == 9
    return transfer, matrix, is_hdr


@node("tonemap", category="Degrade", label="HDR to SDR Tonemap", probability=True,
      description="Converts HDR (PQ or HLG, BT.2020) video to 8-bit BT.709 SDR. Meant for the HR clip before "
                  "the chunk node; SDR input passes through when auto-detect is on.",
      inputs={"clip": "clip"}, outputs={"clip": "clip"},
      params=[
          Param("algorithm", "choice", "hable", choices=ALGORITHMS, random=True, help="Tonemapping curve."),
          Param("desat", "float", 0.5, random=True, help="Desaturation of bright colours.", min=0.0, max=1.0),
          Param("target_nits", "float", 100.0, help="Signal peak handed to the curve, in units of 100 nits "
                                                   "(see the node description).", min=1.0),
          Param("auto_detect", "bool", True, help="Skip clips that are not HDR."),
      ])
def tonemap(ctx, clip, *, algorithm, desat, target_nits, auto_detect):
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown tonemap algorithm {algorithm!r}; known: {', '.join(ALGORITHMS)}")
    transfer, matrix, is_hdr = source_description(clip.get_frame(0))
    if auto_detect and not is_hdr:
        ctx.note("skipped", "SDR input")
        raise NodeSkipped()
    ctx.note("algorithm", algorithm)
    ctx.note("desat", round(desat, 3))
    ctx.note("target_nits", target_nits)
    ctx.note("detected_hdr", is_hdr)

    kwargs = {"transfer_in_s": "std-b67" if transfer == HLG_TRANSFER else "st2084",
              "primaries_in_s": "2020"}
    if clip.format.color_family == vs.YUV:
        kwargs["matrix_in_s"] = "2020cl" if matrix == 10 else "2020ncl"
    linear = core.resize.Bicubic(clip, format=vs.RGBS, transfer_s="linear", primaries_s="709",
                                 nominal_luminance=100.0, **kwargs)
    planes = split_planes(linear)
    mapped = tonemap_planes(*planes, algorithm, float(desat), float(target_nits))
    sdr = core.resize.Bicubic(merge_rgb(mapped), format=vs.YUV420P8, matrix_s="709", transfer_in_s="linear",
                              transfer_s="709", primaries_in_s="709", primaries_s="709", range_s="limited",
                              dither_type="error_diffusion")
    return {"clip": sdr}
