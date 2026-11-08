"""
Splitting a video into chunks. Everything downstream of the chunk node runs
once per chunk, so each chunk gets its own random degradations, the way
video-destroyer degraded each chunk file on its own.
"""

import vapoursynth as vs

from ..registry import Param, node

core = vs.core


@node("chunk", category="Chunk", label="Split into Chunks", chunker=True,
      description="Splits the video by scene, by duration or by frame count. "
                  "A second clip on 'aux' is cut at the same frames.",
      inputs={"clip": "clip", "aux": "clip"}, optional_inputs=("aux",),
      outputs={"clip": "clip", "aux": "clip"},
      params=[
          Param("strategy", "choice", "scene", choices=["scene", "duration", "frames"]),
          Param("chunk_duration", "float", 15.0, help="Seconds per chunk, for 'duration'.", min=0.1),
          Param("frames_per_chunk", "int", 5, help="Frames per chunk, for 'frames'.", min=1),
          Param("threshold", "float", 15.0, help="Scene cut threshold, on PySceneDetect's 0-255 content scale.",
                min=0.0),
          Param("min_duration", "float", 1.0, help="Scenes shorter than this many seconds are dropped.", min=0.0),
          Param("max_chunks", "int", 500, help="Keep at most this many chunks, spread evenly over the video "
                                               "(0 keeps all).", min=0),
          Param("downscale", "int", 12, help="Scene detection looks at the video shrunk by this factor.", min=1),
      ])
def chunk(ctx, clip, aux=None, *, strategy, chunk_duration, frames_per_chunk, threshold, min_duration,
          max_chunks, downscale):
    if aux is not None and aux.num_frames != clip.num_frames:
        raise ValueError(f"the clips to chunk differ in length ({clip.num_frames} and {aux.num_frames} frames); "
                         "align them first")
    fps = clip.fps.numerator / clip.fps.denominator if clip.fps.denominator else 24.0
    total = clip.num_frames

    if strategy == "frames":
        step = int(frames_per_chunk)
        ranges = [(s, min(s + step, total)) for s in range(0, total, step)]
        ranges = [r for r in ranges if r[1] - r[0] == step]
    elif strategy == "duration":
        step = max(1, round(float(chunk_duration) * fps))
        ranges = [(s, min(s + step, total)) for s in range(0, total, step)]
    else:
        cuts = scene_cuts(ctx, clip, float(threshold), int(downscale))
        bounds = [0, *cuts, total]
        min_frames = round(float(min_duration) * fps)
        ranges = [(a, b) for a, b in zip(bounds, bounds[1:]) if b - a >= max(min_frames, 1)]

    if max_chunks and len(ranges) > max_chunks:
        step = len(ranges) / max_chunks
        ranges = [ranges[int(i * step)] for i in range(max_chunks)]
    ctx.message(f"{ctx.video_name}: {len(ranges)} chunks")
    return ranges


def scene_cuts(ctx, clip, threshold, downscale):
    """
    Frames that start a new scene: where the mean per-plane difference from
    the previous frame, in 8-bit YUV of a shrunk copy, passes the threshold.
    This plays the part of PySceneDetect's content detector (also an average
    per-channel delta on a 0-255 scale) without a second decoder.
    """
    width = max(16, (clip.width // downscale) // 2 * 2)
    height = max(16, (clip.height // downscale) // 2 * 2)
    kwargs = {"matrix_s": "709"} if clip.format.color_family == vs.RGB else {}
    small = core.resize.Bilinear(clip, width, height, format=vs.YUV444P8, **kwargs)
    previous = small[0] + small[:-1]
    for plane, prefix in ((0, "Y"), (1, "U"), (2, "V")):
        small = core.std.PlaneStats(small, previous, plane=plane, prop=prefix)

    cuts = []
    for n, frame in enumerate(small.frames(close=True)):
        if n == 0:
            continue
        props = frame.props
        delta = (props["YDiff"] + props["UDiff"] + props["VDiff"]) / 3 * 255
        if delta >= threshold:
            cuts.append(n)
        if n % 2000 == 0:
            ctx.emit({"message": f"{ctx.video_name}: finding scenes, frame {n} of {clip.num_frames}"})
    return cuts
