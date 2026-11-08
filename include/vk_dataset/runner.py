"""
Evaluates a Graph: once per input video, and once per chunk of each video.

Evaluation pulls from the writers. A node's inputs are evaluated first and
each node runs at most once per scope: per-video results are kept for the
whole video, per-chunk results are dropped when the chunk is done. Most
nodes only build VapourSynth filters, so running a node is cheap; the work
happens when a writer requests frames, or when a node such as the codec
round trip renders its input on the spot.

A lazy input (see registry.py) is evaluated only when its node calls it, so
the branches a random choice does not take never run. Which lazy inputs a
node pulled is kept per scope, so the settings a writer records are those of
the nodes that fed it.

Draws are keyed by node, not by evaluation order: a per-chunk node draws
from the job seed, the video, the chunk and its id, a per-video node from
the seed, the video and its id. A branch's nodes draw the same whether or
not the nodes beside them run. A writer with variants evaluates its chunk
scope again per variant, with the variant number added to every per-chunk
key, so variant 0 draws what a job without variants draws.

A setting linked to a value node (links.py) takes the value node's output
for the scope instead of its own value or sampler; the value node runs when
the node it feeds runs, so a node that is off or skipped on a chunk never
pulls its links (its probability excepted, which decides that). A value node
that is off gives nothing, and the settings it feeds keep their own values.
The linked value is recorded like a draw, and so is each value node's output.

A source node whose path is a folder makes the job a batch. Every source
must then name the same number of videos, which are paired by sorted file
name, so a folder of HR releases can be matched with a folder of LR ones.
"""

import gc
import json
import os
import shutil
import sys
import tempfile
import time

import vapoursynth as vs

from .graph import Graph
from .links import LinkError, coerce
from .params import rng_for, sample, is_sampler
from .registry import NODE_TYPES

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".m2ts", ".ts", ".y4m", ".vob", ".mpg")


def emit_json(obj):
    """Writes one protocol line to stdout."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


class NodeSkipped(Exception):
    """Raised by a node that decides it does not apply to this chunk."""


class Context:
    """
    What a node function sees besides its inputs and settings. One Context
    lives for the whole run; the runner moves it from video to video and
    chunk to chunk.
    """

    def __init__(self, graph, emit, work_dir):
        self.graph = graph
        self.core = vs.core
        self.emit = emit
        self.work_dir = work_dir
        self.seed = graph.seed
        self.video_index = None
        self.video_paths = {}
        self.video_name = None
        self.chunk_index = None
        self.chunk_range = None
        self.chunk_count = 0
        self.variant = 0
        self.variants = 1
        self.node = None
        self.rng = None
        self._state = {}
        self._chunk_temp = []
        self._finishers = []
        self.chunk_record = {}
        self.video_record = {}
        # node id -> the lazy input ports it pulled, per scope.
        self.chunk_pulls = {}
        self.video_pulls = {}

    # Values the current node draws for itself (a codec, a filter) go into
    # the chunk's record, which writers copy into the manifest.
    def note(self, key, value):
        self.record_for(self.node).setdefault(self.node.id, {}).setdefault("params", {})[key] = value

    def record_for(self, node):
        return self.chunk_record if node.chunk_scope else self.video_record

    def pulls_for(self, node):
        return self.chunk_pulls if node.chunk_scope else self.video_pulls

    def settings_record(self):
        """What the nodes feeding the current node drew for the current chunk, including the per-video nodes."""
        merged = {**self.video_record, **self.chunk_record}
        upstream = upstream_ids(self.graph, [self.node], {**self.video_pulls, **self.chunk_pulls})
        return {k: v for k, v in merged.items() if k in upstream and (v.get("applied") or v.get("params"))}

    def state(self, node_id=None):
        """A dict that lives for the whole run, private to one node."""
        return self._state.setdefault(node_id or self.node.id, {})

    def temp_path(self, suffix):
        """A file path in the work folder, deleted when the chunk is done."""
        fd, path = tempfile.mkstemp(suffix=suffix, dir=self.work_dir)
        os.close(fd)
        os.remove(path)
        self._chunk_temp.append(path)
        return path

    def on_finish(self, func):
        """Registers func(ctx) to run once after the last video."""
        if func not in self._finishers:
            self._finishers.append(func)

    def message(self, text):
        self.emit({"message": text})

    def enter_video(self, video_index, paths):
        """Moves to a video of the batch, with an empty record and no current chunk."""
        self.video_index = video_index
        self.video_paths = paths
        self.video_name = video_name(paths)
        self.chunk_index = None
        self.chunk_range = None
        self.chunk_record = {}
        self.video_record = {}
        self.chunk_pulls = {}
        self.video_pulls = {}

    def enter_chunk(self, chunk_index, chunk_range):
        """Moves to a chunk of the current video. A chunk_range of None is the whole video of a job without a chunk node."""
        self.chunk_index = chunk_index if chunk_range is not None else None
        self.chunk_range = chunk_range
        self.chunk_record = {}
        self.chunk_pulls = {}
        self.variant = 0


class Lazy:
    """A lazy input: calling it evaluates its source (once per scope) and notes the pull."""

    def __init__(self, ctx, node, port, video_values, chunk_values):
        self.ctx, self.node, self.port = ctx, node, port
        self.video_values, self.chunk_values = video_values, chunk_values

    def __call__(self):
        self.ctx.pulls_for(self.node).setdefault(self.node.id, set()).add(self.port)
        src_id, src_port = self.node.sources[self.port]
        return evaluate(self.ctx, self.ctx.graph.nodes[src_id], self.video_values, self.chunk_values)[src_port]


def upstream_ids(graph, roots, pulls):
    """
    The ids of the given nodes and of every node they read from, following a
    lazy input only when `pulls` (node id -> ports) says it was pulled.
    """
    seen, stack = set(), [n.id for n in roots]
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        node = graph.nodes[node_id]
        for port, (src, _) in node.sources.items():
            if port not in node.type.lazy_inputs or port in pulls.get(node_id, ()):
                stack.append(src)
        stack.extend(src for src, _ in node.links.values())
    return seen


def run(graph, emit=emit_json, work_dir=None):
    """Runs a job. `graph` is a Graph, a dict, or a path to a job file."""
    if isinstance(graph, (str, os.PathLike)):
        graph = Graph.load(graph)
    elif isinstance(graph, dict):
        graph = Graph(graph)

    own_work_dir = work_dir is None
    work_dir = work_dir or graph.settings.get("work_dir") or tempfile.mkdtemp(prefix="vk_dataset_")
    os.makedirs(work_dir, exist_ok=True)
    ctx = Context(graph, emit, work_dir)
    started = time.time()
    try:
        batches = video_batches(graph)
        for video_index, paths in enumerate(batches):
            ctx.enter_video(video_index, paths)
            _run_video(ctx, video_index, len(batches))
        for func in ctx._finishers:
            func(ctx)
        summary = {key: value for state in ctx._state.values() for key, value in state.get("summary", {}).items()}
        return {"videos": len(batches), "seconds": round(time.time() - started, 1), **summary}
    finally:
        if own_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


def _run_video(ctx, video_index, video_count):
    graph = ctx.graph
    video_values = {}
    chunks = chunk_ranges(ctx, video_values)
    if not chunks:
        ctx.message(f"{ctx.video_name}: no chunks found")
        return
    ctx.chunk_count = len(chunks)

    for chunk_index, chunk_range in enumerate(chunks):
        ctx.enter_chunk(chunk_index, chunk_range)
        chunk_values = {}
        try:
            for sink in graph.sinks:
                if sink.chunk_scope or chunk_index == 0:
                    evaluate_sink(ctx, sink, video_values, chunk_values)
        finally:
            chunk_values.clear()
            gc.collect()
            clean_chunk_temp(ctx)
        done = (video_index + (chunk_index + 1) / len(chunks)) / video_count
        ctx.emit({"progress": round(done, 4),
                  "message": f"{ctx.video_name}: chunk {chunk_index + 1} of {len(chunks)}"})


def evaluate_sink(ctx, sink, video_values, chunk_values):
    """
    Pulls a writer through the current chunk, once per variant. Variant 0
    shares the chunk's values with the other writers; every other variant
    evaluates the chunk scope afresh, with its own draws and record.
    """
    ctx.variants = sink.variants
    try:
        evaluate(ctx, sink, video_values, chunk_values)
        if sink.variants > 1:
            shared = ctx.chunk_record, ctx.chunk_pulls
            try:
                for variant in range(1, sink.variants):
                    ctx.variant, ctx.chunk_record, ctx.chunk_pulls = variant, {}, {}
                    evaluate(ctx, sink, video_values, {})
            finally:
                ctx.variant = 0
                ctx.chunk_record, ctx.chunk_pulls = shared
    finally:
        ctx.variants = 1


def chunk_ranges(ctx, video_values):
    """The current video's chunks as (start, end) frame ranges, or [None] when the job has no chunk node."""
    if ctx.graph.chunker is None:
        return [None]
    return evaluate(ctx, ctx.graph.chunker, video_values, None, chunk_list=True)


def evaluate(ctx, node, video_values, chunk_values, chunk_list=False):
    """
    Returns a node's outputs, running it and its inputs as needed. For the
    chunk node with chunk_list set, returns its list of frame ranges instead.
    """
    values = chunk_values if node.chunk_scope and not chunk_list else video_values
    if node.id in values and not chunk_list:
        return values[node.id]

    inputs = {}
    for port, (src_id, src_port) in node.sources.items():
        if port in node.type.lazy_inputs:
            inputs[port] = Lazy(ctx, node, port, video_values, chunk_values)
        else:
            inputs[port] = evaluate(ctx, ctx.graph.nodes[src_id], video_values, chunk_values)[src_port]

    if node.type.chunker:
        cache_key = (node.id, "chunks")
        if cache_key not in video_values:
            video_values[cache_key] = _call(ctx, node, inputs, video_values) if node.enabled else _whole(inputs)
        ranges = video_values[cache_key]
        if chunk_list:
            return ranges
        start, end = ctx.chunk_range
        result = {port: clip[start:end] for port, clip in inputs.items()}
        values[node.id] = result
        return result

    result = _call(ctx, node, inputs, video_values, chunk_values)
    values[node.id] = result
    return result


def rng_key(ctx, node):
    """What a node's generator is seeded from, besides the job seed."""
    if not node.chunk_scope:
        return ctx.video_index, None, node.id
    if ctx.variant:
        return ctx.video_index, ctx.chunk_index, node.id, f"variant{ctx.variant}"
    return ctx.video_index, ctx.chunk_index, node.id


_NOTHING = object()


def linked_value(ctx, node, setting, video_values, chunk_values=None):
    """The value linked into a setting, as the setting takes it, or _NOTHING when its value node is off."""
    src_id, port = node.links[setting]
    outputs = evaluate(ctx, ctx.graph.nodes[src_id], video_values, chunk_values)
    if port not in outputs:
        return _NOTHING
    param = next(p for p in node.type.params + node.type.extra_params() if p.name == setting)
    try:
        return coerce(param, outputs[port], f"{node.label} ({node.id}), {param.title}")
    except LinkError as e:
        raise ValueError(str(e)) from None


def _call(ctx, node, inputs, video_values, chunk_values=None):
    rng = rng_for(ctx.seed, *rng_key(ctx, node))
    probability = float(node.params.get("probability", 1.0)) if node.type.probability else 1.0
    linked = _NOTHING
    if node.type.probability and node.enabled and "probability" in node.links:
        linked = linked_value(ctx, node, "probability", video_values, chunk_values)
        probability = probability if linked is _NOTHING else linked
    applied = node.enabled and (probability >= 1.0 or rng.random() < probability)
    record = ctx.record_for(node).setdefault(node.id, {})
    record["type"] = node.type.name
    record["applied"] = applied
    if not applied:
        return _bypass(node, inputs)

    params = {}
    for param in node.type.params:
        value = node.params.get(param.name, param.default)
        if param.random and is_sampler(value):
            value = sample(value, rng)
            record.setdefault("params", {})[param.name] = value
        params[param.name] = value
    if linked is not _NOTHING:
        record.setdefault("params", {})["probability"] = linked
    for setting in node.links:
        if setting == "probability":
            continue
        value = linked_value(ctx, node, setting, video_values, chunk_values)
        if value is not _NOTHING:
            params[setting] = value
            record.setdefault("params", {})[setting] = value

    # A lazy input runs other nodes from inside this one, so the outer node is put back after.
    outer = ctx.node, ctx.rng
    ctx.node, ctx.rng = node, rng
    try:
        result = node.type.func(ctx, **inputs, **params)
    except NodeSkipped:
        # A node that does not apply drew nothing that shaped the result.
        record["applied"] = False
        record.pop("params", None)
        return _bypass(node, inputs)
    finally:
        ctx.node, ctx.rng = outer
    if node.type.is_value:
        record.setdefault("params", {})["value"] = result["value"]
    if node.type.chunker:
        return result
    return result or {}


def _bypass(node, inputs):
    """A skipped node passes each input to the output of the same name, or its first input to its first output."""
    if not node.type.outputs:
        return {}
    out = {port: inputs[port] for port in node.type.outputs if port in inputs}
    if not out and inputs:
        first_out = next(iter(node.type.outputs))
        out[first_out] = next(iter(inputs.values()))
    return {port: value() if isinstance(value, Lazy) else value for port, value in out.items()}


def _whole(inputs):
    clip = next(iter(inputs.values()))
    return [(0, clip.num_frames)]


def clean_chunk_temp(ctx):
    for path in ctx._chunk_temp:
        for attempt in range(3):
            try:
                if os.path.exists(path):
                    os.remove(path)
                break
            except PermissionError:
                # A source filter on Windows may still hold the file for a
                # moment after its clip is released; the work folder is
                # removed at the end of the run either way.
                gc.collect()
                time.sleep(0.2)
    ctx._chunk_temp.clear()


def list_videos(path):
    """The videos a source path names: the file itself, or a folder's videos in sorted order."""
    if os.path.isdir(path):
        names = sorted(n for n in os.listdir(path) if n.lower().endswith(VIDEO_EXTENSIONS))
        return [os.path.join(path, n) for n in names]
    return [path]


def video_batches(graph):
    """
    One dict per video of the batch, mapping each source node id to its file.
    Sources naming a single file use it for every video of the batch.
    """
    per_source = {}
    for node in graph.order:
        path_param = node.type.param("path")
        if node.type.category != "Source" or path_param is None or path_param.kind != "path":
            continue
        path = node.params.get("path", path_param.default)
        if not path:
            raise ValueError(f"{node.label} ({node.id}) has no file set")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{node.label} ({node.id}): {path} does not exist")
        per_source[node.id] = list_videos(path)
    if not per_source:
        return [{}]
    counts = {len(v) for v in per_source.values() if len(v) != 1}
    if len(counts) > 1:
        detail = ", ".join(f"{k}: {len(v)}" for k, v in per_source.items())
        raise ValueError(f"the sources name different numbers of videos ({detail})")
    count = counts.pop() if counts else 1
    if count == 0:
        raise ValueError("a source folder has no videos in it")
    return [{k: (v[i] if len(v) > 1 else v[0]) for k, v in per_source.items()} for i in range(count)]


def video_name(paths):
    if not paths:
        return "video"
    first = next(iter(paths.values()))
    return os.path.splitext(os.path.basename(first))[0]
