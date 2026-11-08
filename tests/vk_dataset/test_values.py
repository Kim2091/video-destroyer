"""
Value nodes and links: settings driven by a wire from Random Number, Random
Pick, Math or If Equals. Shared draws, derived settings, scopes, the checks
made when a graph loads and when a value arrives, and the manifest records.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "include"))

from vk_dataset import Graph, GraphError, node_specs, run  # noqa: E402
from vk_dataset.links import coerce, LinkError  # noqa: E402
from vk_dataset.registry import NODE_TYPES  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "two_scenes.mkv")


def job(out_dir, values=(), links=(), seed=3, **overrides):
    """Two sharpen nodes in a row on the LR side, with value nodes and links added."""
    nodes = {
        "src": {"type": "load_video", "params": {"path": FIXTURE}},
        "chunks": {"type": "chunk", "params": {"strategy": "frames", "frames_per_chunk": 6}},
        "a": {"type": "sharpen", "params": {}},
        "b": {"type": "sharpen", "params": {}},
        "seq": {"type": "make_sequences", "params": {"length": 5, "max_per_chunk": 1}},
        "out": {"type": "write_dataset", "params": {"directory": str(out_dir)}},
    }
    nodes.update({node_id: {"type": t, "params": dict(p)} for node_id, (t, p) in dict(values).items()})
    for node_id, params in overrides.items():
        nodes[node_id]["params"].update(params)
    return {
        "version": 1, "seed": seed,
        "nodes": [{"id": k, **v} for k, v in nodes.items()],
        "edges": [
            {"from": "src.clip", "to": "chunks.clip"},
            {"from": "chunks.clip", "to": "a.clip"},
            {"from": "a.clip", "to": "b.clip"},
            {"from": "chunks.clip", "to": "seq.hr"},
            {"from": "b.clip", "to": "seq.lr"},
            {"from": "seq.sequences", "to": "out.sequences"},
            *({"from": f, "to": t} for f, t in links),
        ],
    }


def manifest(out_dir):
    with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)["sequences"]


def quiet(_):
    pass


def params_of(entry, node_id):
    return entry["settings"].get(node_id, {}).get("params", {})


def test_specs_group_value_nodes_and_mark_linkable_settings():
    specs = {s["type"]: s for s in node_specs()}
    for name in ("random_number", "random_pick", "math", "if_equals"):
        assert specs[name]["group"] == "Values" and specs[name]["stage"] is None
        assert specs[name]["outputs"] == {"value": "value"} and not specs[name]["inputs"]
    sharpen = {p["name"]: p for p in specs["sharpen"]["params"]}
    assert sharpen["radius"]["linkable"] and sharpen["probability"]["linkable"]
    folders = {p["name"]: p for p in specs["write_dataset"]["params"]}["folders"]
    assert not folders["linkable"]
    # Value nodes sort after every stage.
    assert [s["type"] for s in node_specs()][-4:] == ["if_equals", "math", "random_number", "random_pick"]


def test_one_draw_shared_by_two_nodes(tmp_path):
    raw = job(tmp_path, values={"r": ("random_number", {"low": 0.5, "high": 2.0})},
              links=[("r.value", "a.param:radius"), ("r.value", "b.param:radius")])
    run(raw, emit=quiet)
    entries = manifest(tmp_path)
    radii = [params_of(e, "a")["radius"] for e in entries]
    assert len(entries) > 2 and len(set(radii)) == len(radii), "a new draw per chunk"
    for entry in entries:
        # Sharpen notes its radius to 3 decimals.
        assert params_of(entry, "a")["radius"] == params_of(entry, "b")["radius"]
        assert params_of(entry, "a")["radius"] == pytest.approx(params_of(entry, "r")["value"], abs=1e-3)
        assert 0.5 <= params_of(entry, "a")["radius"] <= 2.0


def test_draws_reproduce_and_follow_the_seed(tmp_path):
    def radii(folder, seed):
        raw = job(folder, values={"r": ("random_number", {"low": 0.5, "high": 2.0})},
                  links=[("r.value", "a.param:radius")], seed=seed)
        run(raw, emit=quiet)
        return [params_of(e, "a")["radius"] for e in manifest(folder)]

    first = radii(tmp_path / "one", 3)
    assert radii(tmp_path / "two", 3) == first
    assert radii(tmp_path / "three", 4) != first


def test_linking_does_not_change_other_draws(tmp_path):
    plain = job(tmp_path / "plain")
    run(plain, emit=quiet)
    linked = job(tmp_path / "linked", values={"r": ("random_number", {"low": 0.5, "high": 2.0})},
                 links=[("r.value", "a.param:radius")])
    run(linked, emit=quiet)
    for x, y in zip(manifest(tmp_path / "plain"), manifest(tmp_path / "linked")):
        assert params_of(x, "a")["amount"] == params_of(y, "a")["amount"]
        assert params_of(x, "b") == params_of(y, "b")


def test_math_derives_a_setting(tmp_path):
    raw = job(tmp_path, values={"r": ("random_number", {"low": 1.0, "high": 2.0}),
                                "half": ("math", {"operation": "multiply", "b": 0.5})},
              links=[("r.value", "a.param:radius"), ("r.value", "half.param:a"), ("half.value", "b.param:radius")])
    run(raw, emit=quiet)
    for entry in manifest(tmp_path):
        assert params_of(entry, "b")["radius"] == pytest.approx(params_of(entry, "a")["radius"] * 0.5, abs=1e-3)


def test_pick_and_if_equals_decide_dropdowns(tmp_path):
    raw = job(tmp_path, values={
        "pick": ("random_pick", {"options": {"choice": ["2d", "horizontal"], "weights": [1, 1]}}),
        "when": ("if_equals", {"equals": "horizontal", "then": 0.3, "otherwise": 3.0}),
    }, links=[("pick.value", "a.param:direction"), ("pick.value", "when.param:value"),
              ("when.value", "b.param:radius")])
    run(raw, emit=quiet)
    seen = set()
    for entry in manifest(tmp_path):
        direction = params_of(entry, "a")["direction"]
        seen.add(direction)
        assert params_of(entry, "b")["radius"] == (0.3 if direction == "horizontal" else 3.0)
    assert seen == {"2d", "horizontal"}


def test_linked_probability_decides_the_skip(tmp_path):
    raw = job(tmp_path, values={"never": ("random_number", {"low": 0.0, "high": 0.0})},
              links=[("never.value", "a.param:probability")])
    run(raw, emit=quiet)
    assert all(not e["settings"].get("a", {}).get("applied", False) for e in manifest(tmp_path))


def test_a_value_node_turned_off_leaves_the_setting_its_own(tmp_path):
    raw = job(tmp_path, values={"r": ("random_number", {"low": 5.0, "high": 6.0})},
              links=[("r.value", "a.param:radius")], a={"radius": 0.9})
    raw["nodes"][-1]["enabled"] = False
    run(raw, emit=quiet)
    for entry in manifest(tmp_path):
        assert params_of(entry, "a")["radius"] == 0.9
        assert "r" not in entry["settings"]


def test_a_skipped_node_does_not_run_its_value_nodes(tmp_path):
    raw = job(tmp_path, values={"r": ("random_number", {"low": 0.5, "high": 2.0})},
              links=[("r.value", "a.param:radius")], a={"probability": 0.0})
    run(raw, emit=quiet)
    assert all("r" not in e["settings"] for e in manifest(tmp_path))


def test_scopes_follow_what_a_value_feeds(tmp_path):
    chunked = Graph(job(tmp_path, values={"r": ("random_number", {"low": 1.0, "high": 5.0, "whole": True})},
                        links=[("r.value", "a.param:radius")]))
    assert chunked.nodes["r"].chunk_scope
    # Feeding the chunk node, which cuts the video once, makes it per video.
    per_video = Graph(job(tmp_path, values={"r": ("random_number", {"low": 6.0, "high": 8.0, "whole": True})},
                          links=[("r.value", "chunks.param:frames_per_chunk"), ("r.value", "a.param:threshold")]))
    assert not per_video.nodes["r"].chunk_scope


def test_a_value_into_the_chunk_node_is_drawn_per_video(tmp_path):
    raw = job(tmp_path, values={"r": ("random_number", {"low": 6.0, "high": 8.0, "whole": True})},
              links=[("r.value", "chunks.param:frames_per_chunk")])
    run(raw, emit=quiet)
    entries = manifest(tmp_path)
    lengths = {e["settings"]["r"]["params"]["value"] for e in entries}
    assert len(lengths) == 1 and lengths <= {6, 7, 8}


@pytest.mark.parametrize("values, links, match", [
    ({"r": ("random_number", {})}, [("r.value", "a.param:nothing")], "no setting 'nothing'"),
    ({"r": ("random_number", {})}, [("r.value", "out.param:folders")], "cannot be linked"),
    ({}, [("chunks.clip", "a.param:radius")], "takes a value"),
    ({"r": ("random_number", {})}, [("r.value", "a.clip")], "is a value, but a.clip takes a clip"),
    ({"r": ("random_number", {}), "s": ("random_number", {})},
     [("r.value", "a.param:radius"), ("s.value", "a.param:radius")], "more than one link"),
    ({"r": ("random_number", {"low": 1.0, "high": 9.0})}, [("r.value", "a.param:radius")], "past its most, 6"),
    ({"r": ("random_number", {"low": 0.0, "high": 1.0})}, [("r.value", "a.param:direction")], "takes one of"),
    ({"p": ("random_pick", {"options": {"choice": ["2d", "diagonal"]}})}, [("p.value", "a.param:direction")],
     "'diagonal', which is not one of its options: mix, 2d, horizontal"),
    ({"p": ("random_pick", {"options": {"choice": []}})}, [("p.value", "a.param:direction")], "no options"),
    ({"p": ("random_pick", {"options": {"choice": ["x"]}})}, [("p.value", "a.param:radius")], "takes a number"),
    ({"m": ("math", {"a": 4.0, "operation": "multiply", "b": 2.0})}, [("m.value", "a.param:radius")],
     "past its most"),
    ({"m": ("math", {}), "n": ("math", {})}, [("m.value", "n.param:a"), ("n.value", "m.param:a")], "cycle"),
])
def test_bad_links_are_refused_on_load(tmp_path, values, links, match):
    with pytest.raises(GraphError, match=match):
        Graph(job(tmp_path, values=values, links=links))


def test_values_through_if_equals_and_math_are_checked_on_load(tmp_path):
    raw = job(tmp_path, values={"r": ("random_number", {"low": 0.0, "high": 1.0}),
                                "when": ("if_equals", {"equals": 5, "then": "2d", "otherwise": "sideways"})},
              links=[("r.value", "when.param:value"), ("when.value", "a.param:direction")])
    with pytest.raises(GraphError, match="'sideways', which is not one of its options"):
        Graph(raw)
    raw = job(tmp_path, values={"p": ("random_pick", {"options": {"choice": ["2d", "sideways"]}}),
                                "m": ("math", {"operation": "divide", "a": 1.0})},
              links=[("p.value", "m.param:b"), ("m.value", "a.param:radius")])
    with pytest.raises(GraphError, match="takes a number"):
        Graph(raw)


def test_a_value_that_cannot_be_told_is_checked_when_it_arrives(tmp_path):
    # 1 / r for r in [-1, 1] has no bounds, so the graph loads, and the first value out of range stops the run.
    raw = job(tmp_path, values={"r": ("random_number", {"low": -1.0, "high": 1.0}),
                                "m": ("math", {"operation": "divide", "a": 1.0})},
              links=[("r.value", "m.param:b"), ("m.value", "a.param:radius")])
    Graph(raw)
    with pytest.raises(ValueError, match=r"Sharpen \(Halos\) \(a\), Radius got -?[0-9.]+, (below its least|past its most)"):
        run(raw, emit=quiet)


def test_division_by_a_range_through_zero_is_checked_at_run_time(tmp_path):
    raw = job(tmp_path, values={"r": ("random_number", {"low": 0.0, "high": 0.0}),
                                "m": ("math", {"operation": "divide", "a": 1.0})},
              links=[("r.value", "m.param:b"), ("m.value", "a.param:radius")])
    Graph(raw)
    with pytest.raises(ValueError, match="divides by 0"):
        run(raw, emit=quiet)


def test_coerce_matches_settings():
    sharpen = NODE_TYPES["sharpen"]
    radius, direction = sharpen.param("radius"), sharpen.param("direction")
    assert coerce(radius, "1.5", "x") == 1.5
    assert coerce(direction, "2d", "x") == "2d"
    scale = NODE_TYPES["scale_to_model"].param("scale")
    assert coerce(scale, 2.0, "x") == 2 and coerce(scale, "3", "x") == 3
    with pytest.raises(LinkError, match="1, 2, 3, 4"):
        coerce(scale, 5, "x")
    frames = NODE_TYPES["chunk"].param("frames_per_chunk")
    assert coerce(frames, 6.6, "x") == 7
    with pytest.raises(LinkError, match="below its least"):
        coerce(frames, 0, "x")


def test_preview_shows_every_value_node(tmp_path):
    from vk_dataset.preview import Preview
    raw = job(tmp_path, values={"r": ("random_number", {"low": 0.5, "high": 2.0}),
                                "loose": ("random_pick", {"options": {"choice": ["x", "y"]}})},
              links=[("r.value", "a.param:radius")])
    preview = Preview(str(tmp_path / "work"))
    try:
        preview.load(raw)
        header, _ = preview.frame("a", "clip", 0, 1, 0)
        assert header["values"]["r"]["value"] == pytest.approx(header["settings"]["a"]["params"]["radius"], abs=1e-3)
        assert header["values"]["loose"]["value"] in ("x", "y")
        # Editing a value node invalidates the nodes it feeds.
        raw["nodes"][-2]["params"]["low"] = 1.9
        preview.load(raw)
        again, _ = preview.frame("a", "clip", 0, 1, 0)
        assert again["settings"]["a"]["params"]["radius"] >= 1.9
    finally:
        preview.close()


def test_if_equals_can_give_a_random_pick(tmp_path):
    raw = job(tmp_path, values={
        "r": ("random_number", {"low": 0.0, "high": 1.0}),
        "when": ("if_equals", {"equals": 5, "then": "2d", "otherwise": {"choice": ["2d", "horizontal"]}}),
    }, links=[("r.value", "when.param:value"), ("when.value", "a.param:direction")])
    run(raw, emit=quiet)
    assert {params_of(e, "a")["direction"] for e in manifest(tmp_path)} == {"2d", "horizontal"}
    raw["nodes"][-1]["params"]["otherwise"] = {"choice": ["2d", "sideways"]}
    with pytest.raises(GraphError, match="'sideways'"):
        Graph(raw)
