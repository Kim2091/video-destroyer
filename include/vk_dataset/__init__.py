"""
vk_dataset - builds paired training datasets from video with VapourSynth.

A dataset job is a graph of nodes saved as JSON (see graph.py). Nodes pass
VapourSynth clips and sequence lists to each other; the runner evaluates the
graph once per video and once per chunk of it, so every chunk draws its own
random degradation settings from a seed and writes them to the manifest.

    python -m vk_dataset run job.json
    python -m vk_dataset nodes

Progress is printed as one JSON object per line, the same protocol as
align_videos.py: {"progress": 0..1, "message": ...} while it works, then
{"result": {...}} or {"error": "..."} last.
"""

import importlib
import pkgutil

from . import nodes
from .registry import NODE_TYPES, node_specs
from .graph import Graph, GraphError
from .runner import run

# Importing a node module registers its node types, so every module in
# nodes/ is imported here and a new one needs no line of its own.
for _module in pkgutil.iter_modules(nodes.__path__):
    importlib.import_module(f"{nodes.__name__}.{_module.name}")

__all__ = ["NODE_TYPES", "node_specs", "Graph", "GraphError", "run"]
