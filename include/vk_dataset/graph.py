"""
A dataset job as saved to disk, checked against the registered node types.

    {
      "version": 1,
      "seed": 1234,
      "nodes": [
        {"id": "src", "type": "load_video", "params": {"path": "in.mkv"}},
        {"id": "down", "type": "resize", "params": {"scale": 0.5}, "enabled": true}
      ],
      "edges": [{"from": "src.clip", "to": "down.clip"}]
    }

An edge into "node.param:<setting>" links a value node's output into that
setting (links.py): {"from": "pick.value", "to": "down.param:scale"}. A
setting takes one link, and a link may only carry a value. A node's links
are kept apart from its inputs, in GraphNode.links, but count for the order
of evaluation and for cycles like any edge.

Every check happens here, before anything runs, so a job that loads is one
the runner can evaluate: every edge joins ports of the same type, every
required input has exactly one edge, there are no cycles, there is at most
one chunk node, and there is at least one node with no outputs (a writer) to
pull the graph through.

A node in chunk scope runs once per chunk: the chunk node itself and
everything that reads from it, directly or not. Every other node runs once
per video. A node that reads a chunk-scope node is in chunk scope too, so a
per-video node can never see a per-chunk value. A writer with variants
repeats its chunk scope, so it must be in one.

A value node runs once per chunk when every node it feeds does (the chunk
node, which cuts the video once, counts as a per-video node here), and once
per video otherwise. So a value always reaches its settings in their scope,
and a value shared by a per-video node and a per-chunk node is the same for
the whole video. What a value node can give is checked against the settings
it feeds where it can be told before running.

A job to run also keeps the pairs registry.py declares: everything a node
with `needs_after` (Telecine) feeds passes through an enabled node of that
type (Bad Telecine Fix) before it becomes sequences or is written, and a
node with `needs_before` has an enabled one of its type upstream. A graph
for previewing part of a job skips this, so a chain can be looked at while
it is being built.
"""

import json

from . import links
from .params import is_sampler
from .registry import NODE_TYPES

GRAPH_VERSION = 1


class GraphError(ValueError):
    pass


class GraphNode:
    def __init__(self, raw, node_type):
        self.id = raw["id"]
        self.type = node_type
        self.params = dict(raw.get("params") or {})
        self.enabled = bool(raw.get("enabled", True))
        self.label = raw.get("label") or node_type.label
        # input port -> (source node id, source port)
        self.sources = {}
        # setting name -> (value node id, its port), for the settings a link drives
        self.links = {}
        self.chunk_scope = False

    @property
    def variants(self):
        """How many times a writer evaluates its chunk per chunk; 1 for every other node."""
        return max(1, int(self.params.get("variants", 1))) if self.type.variants else 1


class Graph:
    def __init__(self, raw, needs_writer=True):
        """needs_writer=False accepts a graph with no writer yet, for previewing part of a job."""
        if not isinstance(raw, dict):
            raise GraphError("a job must be a JSON object")
        version = raw.get("version", GRAPH_VERSION)
        if version != GRAPH_VERSION:
            raise GraphError(f"this job is version {version}; this vk_dataset reads version {GRAPH_VERSION}")
        self.seed = int(raw.get("seed", 0))
        self.settings = dict(raw.get("settings") or {})
        self.nodes = {}
        for item in raw.get("nodes") or []:
            node_id = item.get("id")
            if not node_id or not isinstance(node_id, str):
                raise GraphError("every node needs a string id")
            if node_id in self.nodes:
                raise GraphError(f"two nodes have the id {node_id!r}")
            node_type = NODE_TYPES.get(item.get("type"))
            if node_type is None:
                raise GraphError(f"node {node_id!r} has an unknown type {item.get('type')!r}")
            node = GraphNode(item, node_type)
            known = {p.name for p in node_type.params + node_type.extra_params()}
            unknown = set(node.params) - known - {"probability"}
            if unknown:
                raise GraphError(f"node {node_id!r} has unknown settings: {', '.join(sorted(unknown))}")
            for param in node_type.params:
                value = node.params.get(param.name, param.default)
                if param.kind == "choice" and is_sampler(value):
                    _check_choice_sampler(node_id, param, value)
            self.nodes[node_id] = node

        for edge in raw.get("edges") or []:
            src_id, src_port = _split(edge.get("from"), "from")
            dst_id, dst_port = _split(edge.get("to"), "to")
            src, dst = self._node(src_id), self._node(dst_id)
            if src_port not in src.type.outputs:
                raise GraphError(f"{src.type.label} ({src_id}) has no output {src_port!r}")
            setting = links.param_port(dst_port)
            if setting is not None:
                self._link(src, src_port, dst, setting)
                continue
            if dst_port not in dst.type.inputs:
                raise GraphError(f"{dst.type.label} ({dst_id}) has no input {dst_port!r}")
            if src.type.outputs[src_port] != dst.type.inputs[dst_port]:
                raise GraphError(f"{src_id}.{src_port} is a {src.type.outputs[src_port]}, "
                                 f"but {dst_id}.{dst_port} takes a {dst.type.inputs[dst_port]}")
            if dst_port in dst.sources:
                raise GraphError(f"{dst_id}.{dst_port} has more than one input")
            dst.sources[dst_port] = (src_id, src_port)

        for node in self.nodes.values():
            missing = [p for p in node.type.inputs if p not in node.sources and p not in node.type.optional_inputs]
            if missing:
                raise GraphError(f"{node.label} ({node.id}) needs an input on: {', '.join(missing)}")

        self.order = self._topological_order()

        chunkers = [n for n in self.nodes.values() if n.type.chunker]
        if len(chunkers) > 1:
            raise GraphError("a job can have only one chunk node")
        self.chunker = chunkers[0] if chunkers else None
        for node in self.order:
            node.chunk_scope = node.type.chunker or any(
                self.nodes[src].chunk_scope for src, _ in node.sources.values())
        self._value_scopes()
        self._check_links()

        for node in self.order:
            if node.type.variants and node.variants > 1 and not node.chunk_scope:
                raise GraphError(f"{node.label} ({node.id}) writes variants, which needs a chunk node upstream")

        if needs_writer:
            self._check_pairs()

        self.sinks = [n for n in self.order if not n.type.outputs]
        if needs_writer and not self.sinks:
            raise GraphError("a job needs at least one writer node")

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def _node(self, node_id):
        node = self.nodes.get(node_id)
        if node is None:
            raise GraphError(f"an edge names a node that does not exist: {node_id!r}")
        return node

    def _topological_order(self):
        order, state = [], {}

        def visit(node, path):
            mark = state.get(node.id)
            if mark == "done":
                return
            if mark == "visiting":
                raise GraphError(f"the graph has a cycle through {' -> '.join(path + [node.id])}")
            state[node.id] = "visiting"
            for src_id, _ in [*node.sources.values(), *node.links.values()]:
                visit(self.nodes[src_id], path + [node.id])
            state[node.id] = "done"
            order.append(node)

        for node in self.nodes.values():
            visit(node, [])
        return order

    def _link(self, src, src_port, dst, setting):
        where = f"{dst.label} ({dst.id})"
        param = next((p for p in dst.type.params + dst.type.extra_params() if p.name == setting), None)
        if param is None:
            raise GraphError(f"{where} has no setting {setting!r} to link")
        if param.kind not in links.LINKABLE_KINDS:
            raise GraphError(f"{where}, {param.title}: a {param.kind} setting cannot be linked")
        if src.type.outputs[src_port] != "value":
            raise GraphError(f"{src.id}.{src_port} is a {src.type.outputs[src_port]}, but a setting takes a value")
        if setting in dst.links:
            raise GraphError(f"{where}, {param.title}: has more than one link")
        dst.links[setting] = (src.id, src_port)

    def links_from(self, node_id):
        """(node, setting) for every setting the given node's value is linked into."""
        return [(n, setting) for n in self.order for setting, (src, _) in n.links.items() if src == node_id]

    def _value_scopes(self):
        """Puts each value node in chunk scope when everything it feeds runs per chunk (see above)."""
        for node in reversed(self.order):
            if node.type.is_value:
                node.chunk_scope = self.chunker is not None and all(
                    consumer.chunk_scope and not consumer.type.chunker for consumer, _ in self.links_from(node.id))

    def _check_links(self):
        for node in self.order:
            for setting, (src_id, _) in node.links.items():
                src = self.nodes[src_id]
                param = next(p for p in node.type.params + node.type.extra_params() if p.name == setting)
                where = f"{node.label} ({node.id}), {param.title}"
                try:
                    links.check(param, links.describe(self, src_id), where, f"{src.label} ({src_id})")
                except links.LinkError as e:
                    raise GraphError(str(e)) from None

    def _check_pairs(self):
        consumers = {}
        for node in self.order:
            for src_id, _ in node.sources.values():
                consumers.setdefault(src_id, []).append(node)
        for node in self.order:
            if not node.enabled:
                continue
            closer = node.type.needs_after
            if closer:
                queue, seen = list(consumers.get(node.id, [])), set()
                while queue:
                    reached = queue.pop(0)
                    if reached.id in seen or (reached.type.name == closer and reached.enabled):
                        continue
                    seen.add(reached.id)
                    if not reached.type.outputs or "sequences" in reached.type.outputs.values():
                        raise GraphError(f"{node.label} ({node.id}) reaches {reached.label} ({reached.id}) without a "
                                         f"{NODE_TYPES[closer].label} between them, so its frames no longer line "
                                         "up with HR")
                    queue.extend(consumers.get(reached.id, []))
            opener = node.type.needs_before
            if opener:
                queue, seen = [node], set()
                while queue:
                    current = queue.pop(0)
                    if current.id in seen:
                        continue
                    seen.add(current.id)
                    if current is not node and current.type.name == opener and current.enabled:
                        break
                    queue.extend(self.nodes[src] for src, _ in current.sources.values())
                else:
                    raise GraphError(f"{node.label} ({node.id}) needs a {NODE_TYPES[opener].label} before it")

    def sources_of_type(self, type_name):
        return [n for n in self.order if n.type.name == type_name]


def _split(ref, which):
    if not isinstance(ref, str) or "." not in ref:
        raise GraphError(f"an edge's {which!r} must be 'node.port', got {ref!r}")
    node_id, port = ref.rsplit(".", 1)
    return node_id, port


def _check_choice_sampler(node_id, param, value):
    """A random pick for a dropdown lists some of its choices, with one weight each if weighted."""
    options = value.get("choice")
    if not isinstance(options, list) or not options:
        raise GraphError(f"node {node_id!r}, {param.name}: a random pick needs at least one option")
    strays = [o for o in options if o not in param.choices]
    if strays:
        raise GraphError(f"node {node_id!r}, {param.name}: {', '.join(map(str, strays))} is not one of "
                         f"{', '.join(map(str, param.choices))}")
    weights = value.get("weights")
    if weights is not None and (not isinstance(weights, list) or len(weights) != len(options)):
        raise GraphError(f"node {node_id!r}, {param.name}: a random pick needs one weight per option")
