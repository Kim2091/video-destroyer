"""
The dataset preview: a long-lived process that shows any node's output for
one chunk of one video, as a run would compute it, without writing anything.

    python -m vk_dataset preview

A request names a node's output port, a video of the batch, a chunk of it
and a frame within the chunk. The graph is evaluated with the runner's own
Context, scopes and seeded draws, but pulled from the requested node instead
of from the writers. A writer has no outputs, so it is never upstream of a
previewed port and never runs; temp files go to the preview's own folder.

What a request sees matches a run of the same job:
    - every node draws from the generator a run gives it, which depends on
      the node, the video, the chunk (per-chunk nodes) and the variant, not
      on what else was evaluated, so only the nodes the request pulls run;
    - a Random Branch pulls only the input it picks, and the drawn settings
      list only the nodes that fed the shown output, as a writer's manifest
      does;
    - a variant previews what a writer with variants writes for it;
    - a seed override replaces the job seed for the draws (and the cache
      keys), but the chunk ranges stay those of the job;
    - every value node's value for the chunk comes back with each frame,
      whether or not the shown output reads it, so a value node shows what
      it gives without a picture of its own;
    - a node's ctx.state lasts for one chunk (per-chunk nodes) or one video
      (per-video nodes), so a node that compares against earlier chunks,
      such as the near-duplicate filter, sees only the previewed chunk.

Caches: the chunk ranges per video (scene detection reads the whole video),
the current video's per-video values, and the current chunk's values. A
node such as the codec round trip encodes its chunk once, and scrubbing or
switching between nodes of that chunk reuses it. Moving to another chunk
releases the old chunk's values and deletes its temp files. Loading an
edited job keeps every cached value whose node, settings and upstream are
unchanged (see _signatures).

Protocol
--------
stdin   one JSON object per line, each with a "cmd" and a "seq":
            {"cmd": "load", "job": {...}, "cwd": "folder relative paths are read against"}
            {"cmd": "chunks", "video": 0, "seed": null}
            {"cmd": "frame", "node": "id", "port": "clip", "video": 0, "chunk": 0, "frame": 0,
             "compare": {"node": "id", "port": "clip"}, "view": "hr", "compareView": "lr", "seed": null,
             "variant": 0}
            {"cmd": "ping"} / {"cmd": "close"}
        Read on its own thread: a frame request still waiting when a newer
        one arrives is answered {"type": "superseded"} instead of rendered.
stdout  binary only. Every reply is

            uint32 big-endian header length
            header, UTF-8 JSON, with the request's "seq"
            payload, header["bytes"] of packed RGB24

        A frame's payload is its picture, then the compare picture when one
        was asked for; the header gives the size of each. An error is a
        header {"type": "error", "error": ...}; it ends that request only.
        Everything else the process prints goes to stderr.
"""

import gc
import hashlib
import json
import os
import queue
import shutil
import struct
import sys
import tempfile
import threading
import time
import traceback
import types
from collections import deque

import numpy as np

from .graph import Graph
from .media import frame_array, to_rgb24
from .runner import Context, chunk_ranges, clean_chunk_temp, evaluate, upstream_ids, video_batches, video_name

# Chunk range lists kept across videos and edits; a scene split reads the whole video.
RANGE_CACHE_SIZE = 64
# Messages nodes emit during one request that are sent back with it.
MESSAGE_LIMIT = 20


class PreviewError(ValueError):
    pass


class Scope:
    """What one video, or one chunk of it, has computed: node outputs, drawn settings, lazy pulls and temp files."""

    def __init__(self, key):
        self.key = key
        self.values = {}
        self.record = {}
        self.pulls = {}
        self.temp = []


def _signatures(graph, cwd):
    """
    A hash per node of everything its outputs depend on besides the video and
    the seed: its type, settings, enabled state and the hashes of its inputs
    and of the value nodes linked into its settings.
    A cached output is reused after an edit when its node's hash is unchanged.
    """
    out = {}
    for node in graph.order:
        sources = sorted((port, out[src], src_port) for port, (src, src_port) in node.sources.items())
        sources += sorted((f"param:{name}", out[src], src_port) for name, (src, src_port) in node.links.items())
        text = json.dumps([cwd, node.type.name, node.params, node.enabled, sources], sort_keys=True, default=str)
        out[node.id] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return out


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return str(value)


def _sequences_json(value):
    def item(seq):
        return {"frames": seq.frames, "crops": {name: list(crop) for name, crop in seq.crops.items()},
                "info": seq.info}
    return {
        "clips": {name: {"width": clip.width, "height": clip.height, "frames": clip.num_frames}
                  for name, clip in value.clips.items()},
        "items": [item(s) for s in value.items],
        "rejected": [item(s) for s in value.rejected],
    }


class Preview:
    def __init__(self, work_dir):
        self.work_dir = work_dir
        self.graph = None
        self.batches = []
        self.signatures = {}
        # Node id -> its ctx.state, carried from request to request.
        self.state = {}
        # (video paths, chunk node hash) -> chunk ranges
        self.ranges = {}
        self.video = None
        self.chunk = None
        # id(clip) -> (clip, its RGB24 conversion), so scrubbing reuses one resize node.
        self.rgb = {}
        self.messages = []

    def _emit(self, obj):
        if "message" in obj:
            self.messages = [*self.messages, str(obj["message"])][-MESSAGE_LIMIT:]
        sys.stderr.write(json.dumps(obj, default=_json_default) + "\n")

    # -- jobs ----------------------------------------------------------------

    def load(self, raw, cwd=None):
        """Loads a job, keeping every cached output the edit did not change."""
        if cwd:
            os.chdir(cwd)
        try:
            graph = Graph(raw, needs_writer=False)
            batches = video_batches(graph)
        except Exception:
            self.graph = None
            raise
        signatures = _signatures(graph, os.getcwd())
        same = {node_id for node_id, sig in signatures.items() if self.signatures.get(node_id) == sig}
        for scope in (self.video, self.chunk):
            if scope is not None:
                # Keys that are not node ids (the chunk node's range list) are put back on use.
                scope.values = {k: v for k, v in scope.values.items() if k in same}
                scope.record = {k: v for k, v in scope.record.items() if k in same}
                scope.pulls = {k: v for k, v in scope.pulls.items() if k in same}
        self.state = {k: v for k, v in self.state.items() if k in same}
        self.rgb.clear()
        self.graph, self.batches, self.signatures = graph, batches, signatures
        return {
            "seed": graph.seed,
            "videos": [{"index": i, "name": video_name(paths)} for i, paths in enumerate(batches)],
            "reused": len(same),
        }

    def _require_graph(self):
        if self.graph is None:
            raise PreviewError("no job is loaded")
        return self.graph

    # -- scopes --------------------------------------------------------------

    def _release(self, scope):
        if scope is None:
            return
        scope.values.clear()
        scope.record.clear()
        scope.pulls.clear()
        self.rgb.clear()
        gc.collect()
        clean_chunk_temp(types.SimpleNamespace(_chunk_temp=scope.temp))

    def _reset_state(self, chunk_scope):
        for node in self.graph.nodes.values():
            if node.chunk_scope == chunk_scope:
                self.state.pop(node.id, None)

    def _context(self, video, seed):
        """A Context on `video`, with the video's scope made current."""
        graph = self._require_graph()
        if not 0 <= video < len(self.batches):
            raise PreviewError(f"there is no video {video}; the job has {len(self.batches)}")
        paths = self.batches[video]
        key = (video, seed, tuple(sorted(paths.items())))
        if self.video is None or self.video.key != key:
            self._release(self.chunk)
            self.chunk = None
            self._release(self.video)
            self.video = Scope(key)
            self._reset_state(chunk_scope=False)
        ctx = Context(graph, self._emit, self.work_dir)
        ctx.seed = seed
        ctx._state = self.state
        ctx.enter_video(video, paths)
        ctx.video_record = self.video.record
        ctx.video_pulls = self.video.pulls
        ctx._chunk_temp = self.video.temp
        return ctx

    def _ranges(self, ctx):
        graph = self.graph
        if graph.chunker is None:
            return [None]
        key = (self.video.key[2], self.signatures[graph.chunker.id])
        cached = self.ranges.pop(key, None)
        if cached is not None:
            self.video.values[(graph.chunker.id, "chunks")] = cached
        ranges = chunk_ranges(ctx, self.video.values)
        self.ranges[key] = ranges
        while len(self.ranges) > RANGE_CACHE_SIZE:
            self.ranges.pop(next(iter(self.ranges)))
        return ranges

    def _seed(self, seed):
        return self._require_graph().seed if seed is None else int(seed)

    def chunks(self, video, seed=None):
        """The chunk ranges of a video; null ranges mean the job has no chunk node."""
        self.messages = []
        ctx = self._context(int(video), self._seed(seed))
        ranges = self._ranges(ctx)
        return {"video": video, "name": ctx.video_name, "chunked": self.graph.chunker is not None,
                "chunks": [list(r) if r is not None else None for r in ranges], "messages": self.messages}

    # -- frames --------------------------------------------------------------

    def _port(self, node_id, port):
        node = self._require_graph().nodes.get(node_id)
        if node is None:
            raise PreviewError(f"the job has no node {node_id!r}")
        if port not in node.type.outputs:
            raise PreviewError(f"{node.label} ({node_id}) has no output {port!r}")
        return node, port

    def _value(self, ctx, node, port):
        outputs = evaluate(ctx, node, self.video.values, self.chunk.values)
        try:
            return outputs[port]
        except KeyError:
            raise PreviewError(f"{node.label} ({node.id}) gives nothing on {port!r} for this chunk") from None

    def _picture(self, ctx, node, port, value, view, n):
        """The RGB24 frame of a port's value, and what the header says about it."""
        kind = node.type.outputs[port]
        info = {"node": node.id, "port": port, "kind": kind, "scope": "chunk" if node.chunk_scope else "video"}
        offset = 0
        if kind == "sequences":
            if not value.clips:
                raise PreviewError(f"{node.label} ({node.id}) holds no clips")
            name = view if view in value.clips else next(iter(value.clips))
            clip = value.clips[name]
            info["view"] = name
            info["clips"] = list(value.clips)
        elif kind in ("clip", "mask", "signal"):
            # A signal shows the chain so far, decoded with plain settings (nodes/ntsc_chain.py).
            clip = value.preview if kind == "signal" else value
            # A per-video clip is shown over the chunk's frames, as the chunk node cuts it.
            if not node.chunk_scope and ctx.chunk_range is not None:
                offset = min(ctx.chunk_range[0], max(0, clip.num_frames - 1))
                clip = clip[offset:min(ctx.chunk_range[1], clip.num_frames)]
        else:
            raise PreviewError(f"{node.label} ({node.id}).{port} is a {kind}, which has no picture")
        if clip.num_frames == 0:
            raise PreviewError(f"{node.label} ({node.id}).{port} has no frames in this chunk")

        cached = self.rgb.get(id(clip))
        if cached is None or cached[0] is not clip:
            cached = self.rgb[id(clip)] = (clip, to_rgb24(clip))
        n = max(0, min(int(n), clip.num_frames - 1))
        image = np.ascontiguousarray(frame_array(cached[1].get_frame(n)))
        info.update({"n": n, "frames": clip.num_frames, "offset": offset,
                     "width": image.shape[1], "height": image.shape[0], "format": clip.format.name})
        return info, image

    def frame(self, node, port, video, chunk, frame, compare=None, view=None, compare_view=None, seed=None,
              variant=0):
        """One frame of a node's output for a chunk, with the chunk's draws; returns (header, [images])."""
        started = time.perf_counter()
        self.messages = []
        seed = self._seed(seed)
        targets = [self._port(node, port)]
        if compare:
            targets.append(self._port(compare["node"], compare["port"]))
        video, chunk, variant = int(video), int(chunk), max(0, int(variant or 0))

        ctx = self._context(video, seed)
        ranges = self._ranges(ctx)
        if not ranges:
            raise PreviewError(f"{ctx.video_name} has no chunks")
        if not 0 <= chunk < len(ranges):
            raise PreviewError(f"{ctx.video_name} has no chunk {chunk}; it has {len(ranges)}")
        cached = self.chunk is not None and self.chunk.key == (self.video.key, chunk, variant)
        if not cached:
            self._release(self.chunk)
            self.chunk = Scope((self.video.key, chunk, variant))
            self._reset_state(chunk_scope=True)
        ctx.enter_chunk(chunk, ranges[chunk])
        ctx.variant = variant
        ctx.chunk_record = self.chunk.record
        ctx.chunk_pulls = self.chunk.pulls
        ctx._chunk_temp = self.chunk.temp

        values = [self._value(ctx, n, p) for n, p in targets]
        main, image = self._picture(ctx, *targets[0], values[0], view, frame)
        images = [image]
        if targets[0][0].type.outputs[port] == "sequences":
            main["sequences"] = _sequences_json(values[0])
        if compare:
            # The compare picture is the same frame of the chunk.
            other, compare_image = self._picture(ctx, *targets[1], values[1], compare_view, main["n"])
            main["compare"] = other
            images.append(compare_image)

        values_shown = self._value_nodes(ctx)
        merged = {**ctx.video_record, **ctx.chunk_record}
        upstream = upstream_ids(self.graph, [n for n, _ in targets], {**ctx.video_pulls, **ctx.chunk_pulls})
        drawn = {k: v for k, v in merged.items() if k in upstream}
        header = {
            "type": "frame", **main, "video": video, "chunk": chunk,
            "range": list(ranges[chunk]) if ranges[chunk] is not None else None, "seed": seed, "variant": variant,
            # The manifest's form: what a writer fed by these nodes records for the chunk.
            "settings": {k: v for k, v in drawn.items() if v.get("applied") or v.get("params")},
            "skipped": [k for k, v in drawn.items() if not v.get("applied")],
            "values": values_shown,
            "cached": cached, "messages": self.messages,
            "ms": round((time.perf_counter() - started) * 1000, 1),
            "bytes": sum(i.nbytes for i in images),
        }
        return header, images

    def _value_nodes(self, ctx):
        """Each enabled value node's value for the chunk, {"value": ...} or {"error": ...}, by node id."""
        out = {}
        for node in self.graph.order:
            if not node.type.is_value or not node.enabled:
                continue
            try:
                outputs = evaluate(ctx, node, self.video.values, self.chunk.values)
                out[node.id] = {"value": outputs.get("value")}
            except Exception as e:  # noqa: BLE001 - one bad value must not hide the frame
                out[node.id] = {"error": str(e) or type(e).__name__}
        return out

    def close(self):
        self._release(self.chunk)
        self._release(self.video)
        self.chunk = self.video = None
        self.graph = None
        self.state = {}
        gc.collect()
        shutil.rmtree(self.work_dir, ignore_errors=True)


# -- server ------------------------------------------------------------------

class Channel:
    """The binary reply channel on stdout."""

    def __init__(self, out):
        self.out = out

    def reply(self, header, images=()):
        encoded = json.dumps(header, default=_json_default).encode("utf-8")
        self.out.write(struct.pack(">I", len(encoded)))
        self.out.write(encoded)
        for image in images:
            self.out.write(memoryview(image).cast("B"))
        self.out.flush()


def _read_commands(commands):
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            commands.put(None)
            return
        line = line.strip()
        if line:
            commands.put(line)


def _parse(line):
    if line is None:
        return None
    try:
        command = json.loads(line)
    except ValueError:
        return {"cmd": None, "seq": -1}
    return command if isinstance(command, dict) else {"cmd": None, "seq": -1}


def handle(preview, channel, command):
    """Answers one command; returns False when the server should stop."""
    name, seq = command.get("cmd"), command.get("seq", -1)
    try:
        if name == "load":
            channel.reply({"type": "loaded", "seq": seq, **preview.load(command["job"], command.get("cwd"))})
        elif name == "chunks":
            channel.reply({"type": "chunks", "seq": seq, **preview.chunks(command["video"], command.get("seed"))})
        elif name == "frame":
            header, images = preview.frame(
                command["node"], command["port"], command["video"], command["chunk"], command.get("frame", 0),
                compare=command.get("compare"), view=command.get("view"),
                compare_view=command.get("compareView"), seed=command.get("seed"), variant=command.get("variant", 0))
            channel.reply({**header, "seq": seq}, images)
        elif name == "ping":
            channel.reply({"type": "ok", "seq": seq})
        elif name == "close":
            channel.reply({"type": "ok", "seq": seq})
            return False
        else:
            channel.reply({"type": "error", "seq": seq, "error": f"unknown command {name!r}"})
    except Exception as e:  # noqa: BLE001 - one bad request must not end the session
        traceback.print_exc(file=sys.stderr)
        channel.reply({"type": "error", "seq": seq, "error": str(e) or type(e).__name__})
    return True


def serve():
    # stdout carries frames and nothing else, so print() anywhere goes to stderr.
    out = sys.stdout.buffer
    sys.stdout = sys.stderr
    if sys.platform == "win32":
        import msvcrt
        msvcrt.setmode(out.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)

    channel = Channel(out)
    preview = Preview(tempfile.mkdtemp(prefix="vk_dataset_preview_"))
    commands = queue.Queue()
    threading.Thread(target=_read_commands, args=(commands,), daemon=True, name="vk-dataset-preview-stdin").start()
    waiting = deque()
    try:
        while True:
            if not waiting:
                waiting.append(_parse(commands.get()))
            while True:
                try:
                    waiting.append(_parse(commands.get_nowait()))
                except queue.Empty:
                    break
            command = waiting.popleft()
            if command is None:
                break
            if command.get("cmd") == "frame" and waiting and waiting[0] is not None \
                    and waiting[0].get("cmd") == "frame":
                channel.reply({"type": "superseded", "seq": command.get("seq", -1)})
                continue
            if not handle(preview, channel, command):
                break
    finally:
        preview.close()
    return 0
