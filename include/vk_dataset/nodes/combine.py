"""
Joining branches of a graph: picking one of several per chunk, blending two,
or taking one where a mask is on (masks.py makes masks).

Random Branch takes its inputs lazily (see registry.py), so only the branch
it picks for a chunk runs; a branch's nodes draw the same settings whether
they are picked or not, as each draws from its own id.
"""

import vapoursynth as vs

from ..media import convert_like
from ..registry import Param, node

core = vs.core

BRANCHES = ("in_1", "in_2", "in_3", "in_4")


def same_size(clip, like, what):
    """
    The clip in the format of `like`, which must be the same size. It takes
    the chroma location of `like` too: a branch through ffmpeg may come back
    labelled differently, which the merge filters refuse, and the shift is
    at most half a chroma pixel.
    """
    if (clip.width, clip.height) != (like.width, like.height):
        raise ValueError(f"{what} is {clip.width}x{clip.height} but the base is {like.width}x{like.height}; "
                         "give both the same size")
    return core.std.CopyFrameProps(convert_like(clip, like), like, props=["_ChromaLocation"])


def branch_weights(weights, connected):
    """One weight per connected input, in input order; inputs past the end of the list weigh 0."""
    if not isinstance(weights, (list, tuple)) or not all(isinstance(w, (int, float)) and w >= 0 for w in weights):
        raise ValueError("weights must be a list of numbers of 0 or more, one per input")
    table = {port: float(weights[i]) if i < len(weights) else 0.0 for i, port in enumerate(BRANCHES)}
    picked = {port: table[port] for port in connected if table[port] > 0}
    if not picked:
        raise ValueError("every connected input has weight 0")
    return picked


@node("random_branch", category="Combine", label="Random Branch",
      inputs={port: "clip" for port in BRANCHES}, optional_inputs=BRANCHES[1:], lazy_inputs=BRANCHES,
      outputs={"clip": "clip"},
      params=[
          Param("weights", "weights", [1.0, 1.0, 1.0, 1.0],
                help="How often each input is picked, in input order. Unconnected inputs are never picked."),
      ])
def random_branch(ctx, in_1, in_2=None, in_3=None, in_4=None, *, weights):
    inputs = dict(zip(BRANCHES, (in_1, in_2, in_3, in_4)))
    picked = branch_weights(weights, [port for port, value in inputs.items() if value is not None])
    port = ctx.rng.choices(list(picked), weights=list(picked.values()))[0]
    ctx.note("branch", BRANCHES.index(port) + 1)
    return {"clip": inputs[port]()}


@node("blend", category="Combine", label="Blend",
      inputs={"a": "clip", "b": "clip"}, outputs={"clip": "clip"},
      params=[
          Param("weight", "float", {"uniform": [0.25, 0.75]}, random=True, min=0.0, max=1.0,
                help="How much of B is mixed in: 0 is all A, 1 is all B."),
      ])
def blend(ctx, a, b, *, weight):
    return {"clip": core.std.Merge(a, same_size(b, a, "B"), weight=min(max(float(weight), 0.0), 1.0))}


def mask_for(mask, like, strength):
    """A mask scaled by strength, in the bit depth of `like` and the size of its first plane."""
    if (mask.width, mask.height) != (like.width, like.height):
        mask = core.resize.Bilinear(mask, like.width, like.height)
    fmt = like.format
    if fmt.sample_type == vs.FLOAT:
        return core.std.Expr(mask, f"x {strength!r} *", format=vs.GRAYS)
    peak = (1 << fmt.bits_per_sample) - 1
    gray = core.query_video_format(vs.GRAY, vs.INTEGER, fmt.bits_per_sample, 0, 0)
    return core.std.Expr(mask, f"x {strength!r} * {peak} *", format=gray.id)


@node("masked_merge", category="Combine", label="Masked Merge",
      inputs={"base": "clip", "overlay": "clip", "mask": "mask"}, outputs={"clip": "clip"},
      params=[
          Param("strength", "float", 1.0, random=True, min=0.0, max=1.0,
                help="How much of the overlay shows where the mask is fully on."),
      ])
def masked_merge(ctx, base, overlay, mask, *, strength):
    overlay = same_size(overlay, base, "the overlay")
    weights = mask_for(mask, base, min(max(float(strength), 0.0), 1.0))
    return {"clip": core.std.MaskedMerge(base, overlay, weights, first_plane=True)}
