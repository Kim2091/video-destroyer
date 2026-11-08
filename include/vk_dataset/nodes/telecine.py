"""
Telecine at the studio, and the bad recovery a player or ripper applies
to it later.

telecine      'film' is 3:2 pulldown: every 4 film frames become 5 video
              frames, two of which mix fields of neighbouring film frames
              (AA BB BC CD DD), and the frame rate goes up by 5/4. The
              phase of the 4-frame cycle and the field order are drawn per
              chunk. 'video' is native interlaced video: every frame takes
              its second field from the next frame, as fields captured at
              different moments do, and the frame count stays.
telecine_fix  maps the fields back to one frame per film frame. In each
              cycle one film frame has no clean video frame of its own, and
              field matching has to rebuild it from two frames; where that
              fails (`mixed_frames` of those frames) the frame comes out
              mixed with its previous or next neighbour, fixed per chunk.
              The recovery is then applied to every frame, as a deinterlacer
              would: combed (left woven), blend (lines averaged, which
              halves vertical detail and turns combing into a double image),
              bob (one field kept, the other interpolated from it: soft and
              jaggy) or adaptive (bwdif, which only interpolates where it
              sees motion). The kept field is always the film frame's own,
              so no frame shows a neighbour's picture alone.

Between the two nodes the clip has the pulldown's frames, so it no longer
lines up with HR; graph.py refuses a job where a Telecine reaches sequences
or a writer without a fix after it. Telecine stamps its cadence on every
frame (CADENCE_PROPS), which an ffmpeg round trip keeps (media.round_trip),
so the fix maps back exactly what was done.
"""

import numpy as np
import vapoursynth as vs

from ..framefx import modify_planes, picked
from ..registry import Param, node
from ..runner import NodeSkipped
from .mixes import MIX, draw

core = vs.core

SOURCES = ["film", "video"]
RECOVERIES = {"combed": 0.3, "blend": 0.25, "bob": 0.2, "adaptive": 0.25}
# Fields per film frame in a 2:3 cycle: A A, B B B, C C, D D D.
CYCLE_FIELDS = (2, 3, 2, 3)
CADENCE_PROPS = ("TelecineFrames", "TelecineFilm", "TelecinePhase", "TelecineTFF")


def pulldown_frames(num_frames, film, phase):
    """Per video frame, the film frames of its first and second field in time."""
    if not film:
        return [(n, min(n + 1, num_frames - 1)) for n in range(num_frames)]
    fields = []
    for n in range(num_frames):
        fields += [n] * CYCLE_FIELDS[(n + phase) % 4]
    # An odd last field is paired with a repeat of itself.
    if len(fields) % 2:
        fields.append(fields[-1])
    return list(zip(fields[0::2], fields[1::2]))


def weave(top, bottom, mixed):
    """Top lines from `top` and bottom lines from `bottom` on the frames where `mixed` is true."""
    def func(n, plane, arrays):
        if not mixed[n]:
            return arrays[0]
        out = arrays[0].copy()
        out[1::2] = arrays[1][1::2]
        return out
    return modify_planes(top, range(top.format.num_planes), func, others=[bottom])


def recovered_frames(frames, num_frames, film, tff, partner, mixed_share, seed):
    """
    Per film frame, the video frames its top and bottom lines come from, and
    whether its own field is the top one. A film frame with a clean video
    frame takes it. One whose fields straddle two video frames is rebuilt
    from them, or on a failed match takes the one shared with `partner`
    (-1 the previous film frame, +1 the next). Native video has no clean
    frames: each frame keeps its own first field.
    """
    clean, as_first, as_second = {}, {}, {}
    for k, (first, second) in enumerate(frames):
        if first == second:
            clean.setdefault(first, k)
        else:
            as_first.setdefault(first, k)
            as_second.setdefault(second, k)
    top, bottom, own_top = [], [], []
    for n in range(num_frames):
        if n in clean:
            k = clean[n]
            top.append(k)
            bottom.append(k)
            own_top.append(tff)
        elif film and n in as_first and n in as_second:
            before, after = as_second[n], as_first[n]
            if np.random.default_rng([seed, n]).random() < mixed_share:
                k = before if partner < 0 else after
                top.append(k)
                bottom.append(k)
                own_top.append(tff == (k == after))
            else:
                # The first field of `after` is n's; it is the top one when top is first.
                top.append(after if tff else before)
                bottom.append(before if tff else after)
                own_top.append(tff)
        else:
            k = as_first.get(n, as_second.get(n))
            top.append(k)
            bottom.append(k)
            own_top.append(tff == (n in as_first))
    return top, bottom, own_top


def bob(clip, own_top):
    """Keeps each frame's own field and fills the other lines by averaging the lines around them."""
    def func(n, plane, arrays):
        keep_top = own_top[n]
        src = arrays[0].astype(np.float32)
        out = src.copy()
        start = 1 if keep_top else 0
        above = np.vstack([src[:1], src[:-1]])
        below = np.vstack([src[1:], src[-1:]])
        if keep_top:
            below[-1] = src[-2] if src.shape[0] > 1 else src[-1]
        else:
            above[0] = src[1] if src.shape[0] > 1 else src[0]
        out[start::2] = (above[start::2] + below[start::2]) / 2
        return out if arrays[0].dtype == np.float32 else np.rint(out).astype(arrays[0].dtype)
    return modify_planes(clip, range(clip.format.num_planes), func)


def scaled_fps(clip, num, den):
    """The clip with its frame rate times num/den; a clip without a fixed rate keeps it."""
    if not clip.fps.denominator or not clip.fps.numerator:
        return clip
    return core.std.AssumeFPS(clip, fpsnum=clip.fps.numerator * num, fpsden=clip.fps.denominator * den)


@node("telecine", category="Degrade", label="Telecine", probability=True, needs_after="telecine_fix",
      inputs={"clip": "clip"}, outputs={"clip": "clip"},
      params=[
          Param("source", "choice", "film", choices=SOURCES,
                help="'film' for 3:2 pulldown of 24p, 'video' for native interlaced video."),
      ])
def telecine(ctx, clip, *, source):
    if source not in SOURCES:
        raise ValueError(f"source must be one of {', '.join(SOURCES)}")
    if clip.format.color_family == vs.RGB:
        raise ValueError("telecine works on YUV or gray clips")
    if "TelecineFrames" in clip.get_frame(0).props:
        raise ValueError("this clip is telecined already; put a Bad Telecine Fix between the two Telecines")
    rng = ctx.rng
    tff = rng.random() < 0.8
    phase = rng.randrange(4)
    film = source == "film"
    frames = pulldown_frames(clip.num_frames, film, phase)
    top = [first if tff else second for first, second in frames]
    bottom = [second if tff else first for first, second in frames]
    out = weave(picked(clip, top), picked(clip, bottom), [t != b for t, b in zip(top, bottom)])
    if film:
        out = scaled_fps(out, 5, 4)
        ctx.note("phase", phase)
    ctx.note("field_order", "top" if tff else "bottom")
    ctx.note("frames", len(frames))
    cadence = dict(zip(CADENCE_PROPS, (clip.num_frames, int(film), phase, int(tff))))
    return {"clip": core.std.SetFrameProps(out, **cadence)}


@node("telecine_fix", category="Degrade", label="Bad Telecine Fix", needs_before="telecine",
      inputs={"clip": "clip"}, outputs={"clip": "clip"},
      params=[
          Param("recovery", "choice", MIX, choices=[MIX, *RECOVERIES],
                help="What the player or rip does with the fields."),
          Param("mixed_frames", "float", {"uniform": [0.5, 1.0]}, random=True,
                help="Film: share of the frames field matching fails to rebuild (0 = a clean match).",
                min=0.0, max=1.0),
      ])
def telecine_fix(ctx, clip, *, recovery, mixed_frames):
    props = clip.get_frame(0).props
    # A Telecine skipped on this chunk leaves nothing to fix.
    if "TelecineFrames" not in props:
        raise NodeSkipped()
    if clip.format.color_family == vs.RGB:
        raise ValueError("the telecine fix works on YUV or gray clips")
    num_frames, film, phase, tff = (int(props[name]) for name in CADENCE_PROPS)
    film, tff = bool(film), bool(tff)
    frames = pulldown_frames(num_frames, film, phase)
    if clip.num_frames != len(frames):
        raise ValueError(f"the Telecine made {len(frames)} frames but {clip.num_frames} arrived here; "
                         "the nodes between Telecine and Bad Telecine Fix must keep every frame")
    rng = ctx.rng
    chosen = draw(rng, recovery, RECOVERIES)
    partner = rng.choice([-1, 1])
    seed = rng.getrandbits(32)
    top, bottom, own_top = recovered_frames(frames, num_frames, film, tff, partner, mixed_frames, seed)
    ctx.note("recovery", chosen)
    if film:
        ctx.note("partner", "next" if partner > 0 else "previous")
        ctx.note("mixed_frames", round(mixed_frames, 3))

    combed = weave(picked(clip, top), picked(clip, bottom), [t != b for t, b in zip(top, bottom)])
    combed = core.std.RemoveFrameProps(combed, props=list(CADENCE_PROPS))
    if film:
        combed = scaled_fps(combed, 4, 5)
    if chosen == "combed":
        return {"clip": combed}
    if chosen == "blend":
        return {"clip": core.std.Convolution(combed, matrix=[1, 2, 1], mode="v")}
    if chosen == "bob":
        return {"clip": bob(combed, own_top)}
    keep = [core.bwdif.Bwdif(combed, field=0), core.bwdif.Bwdif(combed, field=1)]
    return {"clip": core.std.FrameEval(combed, lambda n: keep[1 if own_top[n] else 0])}
