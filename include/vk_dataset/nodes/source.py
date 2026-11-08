"""Nodes that bring video into the graph."""

from ..media import load
from ..registry import Param, node


@node("load_video", category="Source", label="Load Video",
      description="Opens a video, or each video of a folder in turn.",
      outputs={"clip": "clip"},
      params=[Param("path", "path", "", help="A video file, or a folder of videos to run as a batch.")])
def load_video(ctx, *, path):
    return {"clip": load(ctx.video_paths.get(ctx.node.id, path))}
