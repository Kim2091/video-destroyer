"""
The node types a graph can use, and what each one declares about itself.

A node type is a function registered with @node. The declaration is what the
UI draws (ports, parameters, defaults) and what graph.py validates against,
so both read the same list and neither keeps rules of its own.

Port types:
    clip        a vs.VideoNode
    sequences   a Sequences object (nodes/sequences.py)
    mask        a vs.VideoNode in GRAYS, 0 (off) to 1 (on), the size of the
                clip it was made from (nodes/masks.py)
    signal      an analogue signal chain, simulated when it is decoded
                (nodes/ntsc_chain.py)
    value       one number, text or bool per chunk (or per video), from a
                value node (nodes/values.py); it goes into another node's
                setting through a link (links.py), not into an input port

Scopes:
    video       evaluated once per input video
    chunk       evaluated once per chunk; a node is in chunk scope when it
                is a chunk node or reads from one, which graph.py works out

Lazy inputs are passed to the node as a Lazy, which evaluates its source
only when called. A node that picks one of several inputs declares them lazy,
so the inputs it does not pick never run.

A value node has a value output and nothing else. Its settings take links
like any other node's, so value nodes chain into each other. It may declare
`describe`, what its value can be before it runs (links.py), so a value that
can never suit the setting it feeds is refused when the graph loads.

Pairs of nodes that undo each other's change of the frame count declare it:
a node with `needs_after` must pass through a node of that type before its
video becomes sequences or is written, and a node with `needs_before` must
have one of that type upstream. graph.py enforces both.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from .docs import NODE_DOCS, STAGE_NAMES, VIDEO_OUT, docs_for
from .links import LINKABLE_KINDS


@dataclass
class Param:
    """
    One setting of a node. `kind` is what the UI shows: float, int, bool,
    str, path, choice (one of `choices`), or json for structured settings.
    A float or int param marked `random` may also be given a sampler instead
    of a value; see params.py. A choice param is always random, and its
    sampler may only list its own choices. A value node's own settings may
    also be `value` (any number, text or bool, shown in the editor of the
    setting the node feeds) or `options` (a choice sampler of such values);
    one marked `gives` holds what the node may give, so the UI edits it with
    the choices of the setting the node feeds. Settings of the kinds
    links.py lists can be linked to a value output.
    """
    name: str
    kind: str
    default: Any = None
    label: str = ""
    help: str = ""
    choices: list = field(default_factory=list)
    random: bool = False
    min: float | None = None
    max: float | None = None
    gives: bool = False

    @property
    def title(self):
        """The name the UI shows for the setting."""
        return self.label or self.name.replace("_", " ").capitalize()

    def __post_init__(self):
        if self.kind == "choice":
            self.random = True

    def spec(self):
        out = {"name": self.name, "kind": self.kind, "default": self.default, "label": self.title,
               "help": self.help, "random": self.random, "linkable": self.kind in LINKABLE_KINDS}
        if self.choices:
            out["choices"] = self.choices
        if self.min is not None:
            out["min"] = self.min
        if self.max is not None:
            out["max"] = self.max
        if self.gives:
            out["gives"] = True
        return out


@dataclass
class NodeType:
    name: str
    func: Callable
    category: str
    label: str
    description: str
    inputs: dict
    outputs: dict
    params: list
    optional_inputs: tuple = ()
    chunker: bool = False
    # A node with a probability is skipped on some chunks; it then passes its
    # first input through to its first output, so the two must share a type.
    probability: bool = False
    lazy_inputs: tuple = ()
    # A writer with variants evaluates everything per-chunk above it that
    # many times per chunk, each time with its own draws (see runner.py).
    variants: bool = False
    needs_after: str | None = None
    needs_before: str | None = None
    describe: Callable | None = None

    @property
    def is_value(self):
        """Whether this is a value node: one whose only output is a value."""
        return bool(self.outputs) and set(self.outputs.values()) == {"value"}

    def extra_params(self):
        """The settings the runner handles for every node of this type, besides its own."""
        extra = []
        if self.probability:
            extra.append(Param("probability", "float", 1.0, help="Chance this node is applied to a chunk.",
                               min=0.0, max=1.0))
        if self.variants:
            extra.append(Param("variants", "int", 1, min=1, max=16,
                               help="LR versions per HR sequence, each with its own random draws."))
        return extra

    def spec(self):
        params = [p.spec() for p in self.extra_params()] + [p.spec() for p in self.params]
        docs = docs_for(self.name) or {}
        port_docs = docs.get("ports", {})
        output_docs = docs.get("outputs", {})

        def describe(name, own):
            label, text = own.get(name) or port_docs.get(name) or (VIDEO_OUT if name == "clip" else (name, ""))
            return {"label": label, "help": text}

        def describe_output(name):
            # An output named like an input is that input after this node: its name, but not its help.
            if name in output_docs or name not in self.inputs:
                return describe(name, output_docs)
            return {"label": describe(name, port_docs)["label"], "help": VIDEO_OUT[1]}

        stage = docs.get("stage")
        return {"type": self.name, "category": self.category, "label": self.label,
                "description": docs.get("summary", self.description), "help": docs.get("help", ""),
                "stage": stage, "stageName": STAGE_NAMES.get(stage, docs.get("group", "")),
                "alsoStages": docs.get("also", []),
                **({"group": docs["group"]} if "group" in docs else {}),
                "inputs": self.inputs, "optionalInputs": list(self.optional_inputs), "outputs": self.outputs,
                "inputInfo": {name: describe(name, port_docs) for name in self.inputs},
                "outputInfo": {name: describe_output(name) for name in self.outputs},
                "params": params, "chunker": self.chunker,
                **({"needsAfter": self.needs_after} if self.needs_after else {}),
                **({"needsBefore": self.needs_before} if self.needs_before else {})}

    def param(self, name):
        return next((p for p in self.params if p.name == name), None)


NODE_TYPES: dict[str, NodeType] = {}


def node(name, *, category, label, description="", inputs=None, outputs=None, params=(),
         optional_inputs=(), chunker=False, probability=False, lazy_inputs=(), variants=False,
         needs_after=None, needs_before=None, describe=None):
    """
    Registers a node type. The function is called as func(ctx, **inputs,
    **params) and returns a dict of its outputs by port name.
    """
    def register(func):
        if name in NODE_TYPES:
            raise ValueError(f"node type {name!r} is registered twice")
        if set(lazy_inputs) - set(inputs or {}):
            raise ValueError(f"node type {name!r} declares lazy inputs it does not have")
        if variants and outputs:
            raise ValueError(f"node type {name!r} has outputs, so it cannot write variants")
        # The wording in docs.py wins over the label given here, so error
        # messages and the UI name a node the same way.
        label_text = (docs_for(name) or {}).get("label", label)
        NODE_TYPES[name] = NodeType(name, func, category, label_text, description or (func.__doc__ or "").strip(),
                                    dict(inputs or {}), dict(outputs or {}), list(params),
                                    tuple(optional_inputs), chunker, probability, tuple(lazy_inputs), variants,
                                    needs_after, needs_before, describe)
        return func
    return register


def node_specs():
    """
    Every node type's declaration, for the UI's node palette, in stage order
    and by label within a stage, then the nodes of a group (docs.py) that
    has no stage. The nodes of a chain stay together in their docs order,
    where the chain's first node sorts.
    """
    order = list(NODE_DOCS)
    first = {}
    for name in order:
        chain = NODE_DOCS[name].get("chain")
        if chain:
            first.setdefault(chain, NODE_DOCS[name]["label"])

    def key(spec):
        chain = (docs_for(spec["type"]) or {}).get("chain")
        if chain is None:
            return spec["stage"] or 99, spec.get("group", ""), spec["label"], 0
        return spec["stage"] or 99, spec.get("group", ""), first[chain], order.index(spec["type"])

    return sorted((t.spec() for t in NODE_TYPES.values()), key=key)


def stages():
    """The stages of a job in order, for the UI to show them."""
    from .docs import STAGES
    return [{"stage": number, "name": name, "help": text} for number, name, text in STAGES]
