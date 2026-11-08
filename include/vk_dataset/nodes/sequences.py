"""
Picking frame sequences out of a chunk, and the object that carries them.

A Sequences value holds named clips ("hr", "lr", ...) and a list of
sequences. Each sequence lists, per clip, the frame numbers to write and,
optionally, a crop. Nodes between the sequence picker and the writer drop
sequences (kept in `rejected`), or replace them with cropped tiles, without
touching the clips.
"""

from dataclasses import dataclass, field, replace

from ..registry import Param, node


@dataclass
class Sequence:
    # clip name -> frame numbers within the chunk's clip
    frames: dict
    # clip name -> (left, top, width, height), or no entry for the whole frame
    crops: dict = field(default_factory=dict)
    # Anything a node wants in the manifest for this sequence (scores, tile index).
    info: dict = field(default_factory=dict)


@dataclass
class Sequences:
    clips: dict
    items: list
    # Sequences that filter nodes dropped, each with its reason in info["rejected"].
    rejected: list = field(default_factory=list)

    def replace(self, items):
        return Sequences(self.clips, list(items), self.rejected)

    def filtered(self, check):
        """check(item) returns None to keep an item, or the reason to reject it."""
        kept, rejected = [], list(self.rejected)
        for item in self.items:
            reason = check(item)
            if reason is None:
                kept.append(item)
            else:
                rejected.append(replace(item, info={**item.info, "rejected": reason}))
        return Sequences(self.clips, kept, rejected)


@node("make_sequences", category="Sequences", label="Pick Sequences",
      description="Cuts the chunk into runs of frames. Every clip gets the same frame numbers, "
                  "except that HR can be limited to the centre frame of each run.",
      inputs={"hr": "clip", "lr": "clip"}, optional_inputs=("lr",),
      outputs={"sequences": "sequences"},
      params=[
          Param("length", "int", 5, help="Frames per sequence.", min=1),
          Param("mode", "choice", "full", choices=["full", "gap"],
                help="'full' takes every run back to back; 'gap' skips frames between runs."),
          Param("time_gap", "float", 0.0, help="For 'gap': seconds skipped between runs.", min=0.0),
          Param("frame_skip", "int", 0, help="For 'gap', when time_gap is 0: frames skipped between runs.", min=0),
          Param("max_per_chunk", "int", 0, help="At most this many sequences per chunk (0 = no limit).", min=0),
          Param("hr_frames", "choice", "all", choices=["all", "center"],
                help="'center' writes only the middle HR frame of each run, as VSR targets expect."),
      ])
def make_sequences(ctx, hr, lr=None, *, length, mode, time_gap, frame_skip, max_per_chunk, hr_frames):
    clips = {"hr": hr}
    if lr is not None:
        clips["lr"] = lr
    total = min(c.num_frames for c in clips.values())
    if lr is not None and lr.num_frames != hr.num_frames:
        ctx.message(f"{ctx.video_name}: HR and LR differ in length ({hr.num_frames} and {lr.num_frames} "
                    f"frames); using the first {total}")

    gap = 0
    if mode == "gap":
        fps = hr.fps.numerator / hr.fps.denominator if hr.fps.denominator else 24.0
        gap = round(time_gap * fps) if time_gap > 0 else int(frame_skip)
    step = length + gap

    items = []
    for start in range(0, total - length + 1, step):
        run = list(range(start, start + length))
        frames = {name: run for name in clips}
        if hr_frames == "center":
            frames["hr"] = [run[len(run) // 2]]
        items.append(Sequence(frames, info={"start": start}))
        if max_per_chunk and len(items) >= max_per_chunk:
            break
    return {"sequences": Sequences(clips, items)}
