"""
Masks: where in the picture something should happen. A mask is a GRAYS clip
the size of the clip it was made from, 0 where it is off and 1 where it is
on, with soft values between. Masked Merge (combine.py) uses one to apply a
branch only there, such as banding in flat areas or noise in the shadows.

A mask's settings are drawn per chunk; its values come from the picture,
frame by frame, so it follows the content as real damage does.
"""

import vapoursynth as vs

from ..registry import Param, node

core = vs.core


def luma(clip):
    """The clip's luma as GRAYS, 0 (black) to 1 (white). Integer YUV is read as limited range, as video is."""
    fmt = clip.format
    if fmt.color_family == vs.RGB:
        return core.resize.Bicubic(clip, format=vs.GRAYS, matrix_s="709")
    if fmt.color_family == vs.GRAY:
        if fmt.sample_type == vs.FLOAT:
            return clip
        return core.std.Expr(clip, f"x {(1 << fmt.bits_per_sample) - 1} /", format=vs.GRAYS)
    plane = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    if fmt.sample_type == vs.FLOAT:
        return plane
    shift = fmt.bits_per_sample - 8
    return core.std.Expr(plane, f"x {16 << shift} - {219 << shift} / 0 max 1 min", format=vs.GRAYS)


def ramp(value_expr, low, high):
    """An Expr that is 0 below low, 1 above high, and linear between."""
    width = max(float(high) - float(low), 1e-6)
    return f"{value_expr} {float(low)!r} - {width!r} / 0 max 1 min"


def shape(mask, expand, feather):
    """Grows the mask by `expand` pixels, then softens its edge over `feather` pixels."""
    for _ in range(int(expand)):
        mask = core.std.Maximum(mask)
    if feather:
        mask = core.std.BoxBlur(mask, hradius=int(feather), vradius=int(feather))
    return mask


SHAPE_PARAMS = [
    Param("expand", "int", 0, min=0, max=16, help="Grows the mask by this many pixels."),
    Param("feather", "int", 1, min=0, max=16, help="Softens the mask's edge over this many pixels."),
]


@node("luma_mask", category="Mask", label="Brightness Mask",
      inputs={"clip": "clip"}, outputs={"mask": "mask"},
      params=[
          Param("low", "float", 0.0, random=True, min=0.0, max=1.0, help="Darkest brightness that is on (0 to 1)."),
          Param("high", "float", 0.25, random=True, min=0.0, max=1.0,
                help="Brightest brightness that is on (0 to 1)."),
          Param("softness", "float", 0.08, min=0.0, max=1.0,
                help="How far beyond low and high the mask fades out."),
          *SHAPE_PARAMS,
      ])
def luma_mask(ctx, clip, *, low, high, softness, expand, feather):
    soft = max(float(softness), 1e-6)
    rise = ramp("x", float(low) - soft, float(low)) if float(low) > 0 else "1"
    fall = f"1 {ramp('x', float(high), float(high) + soft)} -" if float(high) < 1 else "1"
    return {"mask": shape(core.std.Expr(luma(clip), f"{rise} {fall} min"), expand, feather)}


@node("edge_mask", category="Mask", label="Edge Mask",
      inputs={"clip": "clip"}, outputs={"mask": "mask"},
      params=[
          Param("threshold", "float", 0.1, random=True, min=0.0, max=2.0,
                help="Edge strength (Sobel, on 0-1 luma) where the mask is fully on; half of it is off."),
          *SHAPE_PARAMS,
      ])
def edge_mask(ctx, clip, *, threshold, expand, feather):
    edges = core.std.Sobel(luma(clip))
    on = max(float(threshold), 1e-6)
    return {"mask": shape(core.std.Expr(edges, ramp("x", on / 2, on)), expand, feather)}


@node("flat_mask", category="Mask", label="Flat Area Mask",
      inputs={"clip": "clip"}, outputs={"mask": "mask"},
      params=[
          Param("threshold", "float", 0.02, random=True, min=0.0, max=1.0,
                help="Local detail (standard deviation, on 0-1 luma) below which an area counts as flat."),
          Param("radius", "int", 3, min=1, max=16, help="Size of the area the detail is measured over."),
          *SHAPE_PARAMS,
      ])
def flat_mask(ctx, clip, *, threshold, radius, expand, feather):
    y = luma(clip)
    r = int(radius)
    mean = core.std.BoxBlur(y, hradius=r, vradius=r)
    square = core.std.BoxBlur(core.std.Expr(y, "x x *"), hradius=r, vradius=r)
    flat = max(float(threshold), 1e-6)
    deviation = "y x x * - 0 max sqrt"
    mask = core.std.Expr([mean, square], f"1 {ramp(deviation, flat, flat * 2)} -")
    return {"mask": shape(mask, expand, feather)}


@node("motion_mask", category="Mask", label="Motion Mask",
      inputs={"clip": "clip"}, outputs={"mask": "mask"},
      params=[
          Param("threshold", "float", 0.04, random=True, min=0.0, max=1.0,
                help="Change from the previous frame (on 0-1 luma) where the mask is fully on; half of it is off."),
          Param("expand", "int", 2, min=0, max=16, help="Grows the mask by this many pixels."),
          Param("feather", "int", 2, min=0, max=16, help="Softens the mask's edge over this many pixels."),
      ])
def motion_mask(ctx, clip, *, threshold, expand, feather):
    y = luma(clip)
    # The first frame is compared with the second, so it is not all still.
    previous = (y[1] + y[:-1]) if y.num_frames > 1 else y
    on = max(float(threshold), 1e-6)
    mask = core.std.Expr([y, previous], ramp("x y - abs", on / 2, on))
    return {"mask": shape(mask, expand, feather)}


MASK_OPS = {
    "invert": "1 x -",
    "both": "x y min",
    "either": "x y max",
    "a_not_b": "x 1 y - min",
}


@node("mask_math", category="Mask", label="Combine Masks",
      inputs={"a": "mask", "b": "mask"}, optional_inputs=("b",), outputs={"mask": "mask"},
      params=[
          Param("operation", "choice", "invert", choices=list(MASK_OPS),
                help="'invert' flips A; 'both' is on where A and B are; 'either' where A or B is; "
                     "'a_not_b' where A is and B is not."),
          Param("expand", "int", 0, min=0, max=16, help="Grows the mask by this many pixels."),
          Param("feather", "int", 0, min=0, max=16, help="Softens the mask's edge over this many pixels."),
      ])
def mask_math(ctx, a, b=None, *, operation, expand, feather):
    if operation not in MASK_OPS:
        raise ValueError(f"unknown operation {operation!r}; known: {', '.join(MASK_OPS)}")
    if operation == "invert":
        mask = core.std.Expr(a, MASK_OPS["invert"])
    else:
        if b is None:
            raise ValueError(f"'{operation}' needs a mask on B")
        if (b.width, b.height) != (a.width, a.height):
            b = core.resize.Bilinear(b, a.width, a.height)
        mask = core.std.Expr([a, b], MASK_OPS[operation])
    return {"mask": shape(mask, expand, feather)}
