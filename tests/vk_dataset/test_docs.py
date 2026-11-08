"""
Every node type is documented, and every example workflow loads, follows the
stage order, and (for the ones that work from one video) runs on the fixture.
"""

import copy
import glob
import json
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "..", "include"))

from vk_dataset import NODE_TYPES, Graph, node_specs, run  # noqa: E402
from vk_dataset.docs import NODE_DOCS, STAGE_NAMES  # noqa: E402

PRESETS = sorted(glob.glob(os.path.join(HERE, "..", "..", "include", "vk_dataset", "presets", "*.json")))
FIXTURE = os.path.join(HERE, "fixtures", "two_scenes.mkv")


def test_every_node_type_is_documented():
    assert set(NODE_DOCS) == set(NODE_TYPES)
    for name, docs in NODE_DOCS.items():
        # A node with no stage, such as a value node, is listed under its group.
        assert docs["stage"] in STAGE_NAMES or (docs["stage"] is None and docs.get("group")), name
        assert docs["label"] and docs["summary"] and docs["help"], name
        node_type = NODE_TYPES[name]
        assert set(docs.get("ports", {})) <= set(node_type.inputs) | set(node_type.outputs), name
        assert set(node_type.inputs) <= set(docs.get("ports", {})), f"{name}: undocumented input"


def test_specs_carry_the_docs_in_stage_order():
    specs = node_specs()
    assert [s["stage"] or 99 for s in specs] == sorted(s["stage"] or 99 for s in specs)
    labels = [s["label"] for s in specs]
    assert len(labels) == len(set(labels))
    for spec in specs:
        assert spec["help"] and spec["stageName"]
        assert all(info["label"] for info in spec["inputInfo"].values())
        assert all(info["label"] for info in spec["outputInfo"].values())
        # A pass-through output keeps its input's name but describes the result, not the input.
        for name, info in spec["outputInfo"].items():
            if name in spec["inputInfo"] and spec["inputInfo"][name]["help"]:
                assert info["help"] != spec["inputInfo"][name]["help"], (spec["type"], name)


@pytest.mark.parametrize("path", PRESETS, ids=os.path.basename)
def test_presets_are_documented_and_laid_out(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["name"] and raw["description"] and raw["summary"]
    for item in raw["nodes"]:
        assert "ui" in item and "note" in item, item["id"]
        if item["type"] == "load_video":
            item["params"]["path"] = FIXTURE
    graph = Graph(raw)
    # Every edge runs forward through the stages, counting a node's `also`
    # stages as places it may sit.
    def stages(node):
        docs = NODE_DOCS[node.type.name]
        return [docs["stage"], *docs.get("also", [])]

    for node in graph.nodes.values():
        for src_id, _ in node.sources.values():
            src = graph.nodes[src_id]
            assert min(stages(src)) <= max(stages(node)), (src.id, node.id)


def node_types(path):
    with open(path, encoding="utf-8") as f:
        return [n["type"] for n in json.load(f)["nodes"]]


# Jobs that cut sequences from one video; the still-shot jobs need still shots the fixture lacks.
SINGLE_VIDEO = [p for p in PRESETS if "make_sequences" in node_types(p)
                and [t for t in node_types(p) if NODE_TYPES[t].category == "Source"] == ["load_video"]]


@pytest.mark.parametrize("path", SINGLE_VIDEO, ids=os.path.basename)
def test_single_video_presets_run(tmp_path, path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    raw = copy.deepcopy(raw)
    if any(item["type"].startswith("ntsc_") for item in raw["nodes"]):
        from vk_dataset.nodes.ntsc_chain import PLUGIN
        if not os.path.exists(PLUGIN):
            pytest.skip("the analogue video plugin is not built (npm run build:native)")
    for item in raw["nodes"]:
        if item["type"] == "load_video":
            item["params"]["path"] = FIXTURE
        if item["type"] == "write_dataset":
            item["params"]["directory"] = str(tmp_path)
        if item["type"] == "chunk":
            item["params"]["min_duration"] = 0.5
        if item["type"] == "tile":
            # The fixture is 320x240, smaller than the presets' tiles.
            item["params"].update(tile_width=128, tile_height=128)
    result = run(raw, emit=lambda _: None)
    assert result.get("sequences", 0) > 0
