"""
Any ffmpeg filter as a degradation. The clip goes through ffmpeg and a
lossless intermediate file, so the damage is exactly what ffmpeg's filter
makes of it and nothing else is added.
"""

import vapoursynth as vs

from ..media import round_trip
from ..registry import Param, node

core = vs.core


def run_filter(ctx, clip, filter_string):
    """
    Runs an ffmpeg -vf filter graph over the clip and returns the result in
    the clip's format, with the clip's frame count. The intermediate is FFV1,
    all intra, so frames decode in any order.
    """
    args = ["-vf", filter_string, "-c:v", "ffv1", "-level", "3", "-g", "1"]
    return round_trip(clip, ctx.temp_path(".mkv"), args)


@node("ffmpeg_filter", category="Degrade", label="ffmpeg Filter", probability=True,
      description="Applies an ffmpeg -vf filter string through a lossless round trip, so any ffmpeg "
                  "degradation can be reproduced exactly. A filter that changes the frame count is "
                  "trimmed or padded with the last frame to the clip's length.",
      inputs={"clip": "clip"}, outputs={"clip": "clip"},
      params=[Param("filter", "str", "", help="The filter graph, as written after -vf.")])
def ffmpeg_filter(ctx, clip, *, filter):
    if not filter.strip():
        raise ValueError("the ffmpeg filter node needs a filter string")
    ctx.note("filter", filter)
    return {"clip": run_filter(ctx, clip, filter)}
