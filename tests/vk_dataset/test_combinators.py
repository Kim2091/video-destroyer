"""
Tests for graph combinators: lazy inputs and Random Branch, Blend, the masks
and Masked Merge, writer variants and subfolders, and the example jobs that
use them (10 and up).
"""

import copy
import glob
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "..", "include"))

from vk_dataset import NODE_TYPES, Graph, GraphError, run  # noqa: E402
from vk_dataset.preview import Preview  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "two_scenes.mkv")
# The examples that work from one video; 14 needs two releases.
PRESETS = sorted(glob.glob(os.path.join(HERE, "..", "..", "include", "vk_dataset", "presets", "1[0-3]-*.json")))
# HR clean-up nodes and the vs-jetpack module each needs.
CLEAN_UP = {"fine_dehalo": "vsdehalo", "fast_line_darken": "vsrgtools"}


def quiet(_):
    pass


def manifest(out_dir):
    with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)["sequences"]


def read_rgb(path):
    import cv2
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def branch_job(out_dir, seed=7, weights=(1, 1)):
    """Two LR branches, a blurred and a noisy one, joined by a Random Branch."""
    nodes = {
        "src": ("load_video", {"path": FIXTURE}),
        "chunks": ("chunk", {"strategy": "frames", "frames_per_chunk": 12}),
        "down": ("resample", {"size": "scale", "scale": 0.5, "kernel": "bicubic"}),
        "soft": ("optics_blur", {}),
        "grain": ("film_grain", {}),
        "pick": ("random_branch", {"weights": list(weights)}),
        "seq": ("make_sequences", {"length": 5, "max_per_chunk": 1}),
        "out": ("write_dataset", {"directory": str(out_dir)}),
    }
    edges = [("src.clip", "chunks.clip"), ("chunks.clip", "down.clip"), ("down.clip", "soft.clip"),
             ("down.clip", "grain.clip"), ("soft.clip", "pick.in_1"), ("grain.clip", "pick.in_2"),
             ("chunks.clip", "seq.hr"), ("pick.clip", "seq.lr"), ("seq.sequences", "out.sequences")]
    return {"version": 1, "seed": seed,
            "nodes": [{"id": k, "type": t, "params": p} for k, (t, p) in nodes.items()],
            "edges": [{"from": a, "to": b} for a, b in edges]}


@pytest.fixture
def preview(tmp_path):
    work = tmp_path / "preview_work"
    work.mkdir()
    p = Preview(str(work))
    yield p
    p.close()


@pytest.fixture
def counted(monkeypatch):
    """Counts the calls of each node type's function."""
    calls = {}

    def wrap(name):
        func = NODE_TYPES[name].func

        def counting(*args, **kwargs):
            calls[name] = calls.get(name, 0) + 1
            return func(*args, **kwargs)
        monkeypatch.setattr(NODE_TYPES[name], "func", counting)

    for name in ("optics_blur", "film_grain"):
        wrap(name)
    return calls


def test_random_branch_runs_only_the_picked_branch(tmp_path, counted):
    run(branch_job(tmp_path), emit=quiet)
    entries = manifest(tmp_path)
    picks = [e["settings"]["pick"]["params"]["branch"] for e in entries]
    assert len(entries) == 12 and set(picks) == {1, 2}
    assert counted == {"optics_blur": picks.count(1), "film_grain": picks.count(2)}
    for entry, pick in zip(entries, picks):
        # The manifest names only the branch that fed the LR.
        assert ("soft" in entry["settings"]) == (pick == 1)
        assert "grain" not in entry["settings"] or pick == 2


def test_branch_draws_do_not_depend_on_siblings(tmp_path):
    run(branch_job(tmp_path / "both"), emit=quiet)
    run(branch_job(tmp_path / "one", weights=(1, 0)), emit=quiet)
    both, one = manifest(tmp_path / "both"), manifest(tmp_path / "one")
    for a, b in zip(both, one):
        assert b["settings"]["pick"]["params"]["branch"] == 1
        if a["settings"]["pick"]["params"]["branch"] == 1:
            assert a["settings"]["soft"] == b["settings"]["soft"]


def test_random_branch_weights_and_bypass(tmp_path):
    raw = branch_job(tmp_path, weights=(0, 0))
    with pytest.raises(ValueError, match="weight 0"):
        run(raw, emit=quiet)
    raw = branch_job(tmp_path / "off")
    raw["nodes"][5]["enabled"] = False
    run(raw, emit=quiet)
    assert all("pick" not in e["settings"] and "soft" in e["settings"] for e in manifest(tmp_path / "off"))


def test_preview_matches_a_branching_run(tmp_path, preview):
    out = tmp_path / "run"
    run(branch_job(out), emit=quiet)
    entries = manifest(out)
    preview.load(branch_job(tmp_path / "never"))
    for show, entry in enumerate(entries[:4], start=1):
        chunk = entry["chunk"]
        # Previewing a branch directly runs it, but the pick's settings still list only what fed it.
        preview.frame("soft", "clip", 0, chunk, 0)
        preview.frame("grain", "clip", 0, chunk, 0)
        header, (image,) = preview.frame("pick", "clip", 0, chunk, 2)
        assert (image == read_rgb(os.path.join(out, "LR", f"show{show:05d}_Frame00003.png"))).all()
        header, _ = preview.frame("seq", "sequences", 0, chunk, 0)
        assert header["settings"] == {k: v for k, v in entry["settings"].items() if k != "out"}


def mask_job(out_dir, mask_type, mask_params=None, strength=1.0):
    """LR is the downscaled chunk, with a heavily blurred copy merged in where the mask is on."""
    nodes = {
        "src": ("load_video", {"path": FIXTURE}),
        "chunks": ("chunk", {"strategy": "frames", "frames_per_chunk": 72}),
        "down": ("resample", {"size": "scale", "scale": 0.5, "kernel": "bicubic"}),
        "soft": ("optics_blur", {"shape": "gaussian", "sigma": 4.0}),
        "mask": (mask_type, mask_params or {}),
        "merge": ("masked_merge", {"strength": strength}),
        "seq": ("make_sequences", {"length": 5, "max_per_chunk": 1}),
        "out": ("write_dataset", {"directory": str(out_dir)}),
    }
    edges = [("src.clip", "chunks.clip"), ("chunks.clip", "down.clip"), ("down.clip", "soft.clip"),
             ("down.clip", "mask.clip"), ("down.clip", "merge.base"), ("soft.clip", "merge.overlay"),
             ("mask.mask", "merge.mask"), ("chunks.clip", "seq.hr"), ("merge.clip", "seq.lr"),
             ("seq.sequences", "out.sequences")]
    return {"version": 1, "seed": 3,
            "nodes": [{"id": k, "type": t, "params": p} for k, (t, p) in nodes.items()],
            "edges": [{"from": a, "to": b} for a, b in edges]}


@pytest.mark.parametrize("mask_type", ["luma_mask", "edge_mask", "flat_mask", "motion_mask"])
def test_masks_preview_as_pictures_and_steer_the_merge(tmp_path, preview, mask_type):
    preview.load(mask_job(tmp_path / "never", mask_type))
    header, (mask,) = preview.frame("mask", "mask", 0, 0, 3)
    assert header["kind"] == "mask" and mask.shape == (120, 160, 3)
    assert (mask[..., 0] == mask[..., 1]).all()
    _, (base,) = preview.frame("down", "clip", 0, 0, 3)
    _, (merged,) = preview.frame("merge", "clip", 0, 0, 3)
    _, (soft,) = preview.frame("soft", "clip", 0, 0, 3)
    # Chroma is subsampled, so only pixels well inside an off or on area are compared.
    off, on = inside(mask[..., 0] == 0), inside(mask[..., 0] == 255)
    assert off.sum() > 100 or on.sum() > 100
    if off.any():
        assert np.abs(merged.astype(int) - base.astype(int))[off].mean() < 1
    if on.any():
        assert np.abs(merged.astype(int) - soft.astype(int))[on].mean() < 1


def inside(region, margin=3):
    """The pixels of a region whose neighbours within `margin` are all in it too."""
    out = region.copy()
    for dy in range(-margin, margin + 1):
        for dx in range(-margin, margin + 1):
            out &= np.roll(region, (dy, dx), axis=(0, 1))
    return out


def test_luma_mask_picks_the_shadows(tmp_path, preview):
    preview.load(mask_job(tmp_path / "never", "luma_mask", {"low": 0.0, "high": 0.3, "feather": 0}))
    _, (mask,) = preview.frame("mask", "mask", 0, 0, 0)
    _, (base,) = preview.frame("down", "clip", 0, 0, 0)
    luma = base @ np.array([0.2126, 0.7152, 0.0722])
    assert mask[luma < 0.25 * 255, 0].min() >= 250 and mask[luma > 0.45 * 255, 0].max() <= 5


def test_mask_math_and_blend(tmp_path, preview):
    raw = mask_job(tmp_path / "never", "luma_mask", {"feather": 0})
    raw["nodes"] += [{"id": "flip", "type": "mask_math", "params": {"operation": "invert"}},
                     {"id": "mix", "type": "blend", "params": {"weight": 0.5}}]
    raw["edges"] += [{"from": "mask.mask", "to": "flip.a"},
                     {"from": "down.clip", "to": "mix.a"}, {"from": "soft.clip", "to": "mix.b"}]
    preview.load(raw)
    _, (mask,) = preview.frame("mask", "mask", 0, 0, 0)
    _, (flipped,) = preview.frame("flip", "mask", 0, 0, 0)
    assert (np.abs(mask.astype(int) + flipped.astype(int) - 255) <= 1).all()
    preview.frame("mix", "clip", 0, 0, 0)
    # Compared as the clips' own planes: RGB clips saturated colours.
    values = preview.chunk.values
    base, soft, mixed = (np.asarray(values[n]["clip"].get_frame(0)[0], dtype=float) for n in ("down", "soft", "mix"))
    assert np.abs(mixed - (base + soft) / 2).max() <= 1


def test_masks_only_join_mask_ports(tmp_path):
    raw = mask_job(tmp_path, "luma_mask")
    raw["edges"][4] = {"from": "mask.mask", "to": "merge.base"}
    with pytest.raises(GraphError, match="mask"):
        Graph(raw)


def test_variants_write_fresh_lr_per_copy(tmp_path):
    run(branch_job(tmp_path / "one"), emit=quiet)
    raw = branch_job(tmp_path / "three")
    raw["nodes"][-1]["params"]["variants"] = 3
    run(raw, emit=quiet)
    one, three = manifest(tmp_path / "one"), manifest(tmp_path / "three")
    assert len(three) == 3 * len(one)
    assert [e["variant"] for e in three[:3]] == [0, 1, 2]
    # Variant 0 is the job without variants; the others draw anew but share the HR frames.
    assert [e["settings"] for e in three if e["variant"] == 0] == [e["settings"] for e in one]
    first = [e for e in three if e["chunk"] == 0]
    assert first[0]["frames"] == first[1]["frames"] and first[0]["settings"] != first[1]["settings"]


def test_variants_preview_like_the_run(tmp_path, preview):
    out = tmp_path / "run"
    raw = branch_job(out)
    raw["nodes"][-1]["params"]["variants"] = 2
    run(raw, emit=quiet)
    entry = next(e for e in manifest(out) if e["variant"] == 1)
    preview.load(branch_job(tmp_path / "never"))
    header, _ = preview.frame("seq", "sequences", 0, entry["chunk"], 0, variant=1)
    assert header["variant"] == 1
    assert header["settings"] == {k: v for k, v in entry["settings"].items() if k != "out"}


def test_variants_need_chunks(tmp_path):
    raw = branch_job(tmp_path)
    raw["nodes"][-1]["params"]["variants"] = 2
    raw["nodes"] = [n for n in raw["nodes"] if n["id"] != "chunks"]
    raw["edges"] = [e for e in raw["edges"] if "chunks" not in e["from"] + e["to"]]
    raw["edges"] += [{"from": "src.clip", "to": "down.clip"}, {"from": "src.clip", "to": "seq.hr"}]
    with pytest.raises(GraphError, match="variants"):
        Graph(raw)


def test_writers_share_a_folder_and_record_their_own_branch(tmp_path):
    raw = branch_job(tmp_path)
    raw["nodes"][-1]["params"]["subfolder"] = "a"
    raw["nodes"] += [{"id": "seq_b", "type": "make_sequences", "params": {"length": 5, "max_per_chunk": 1}},
                     {"id": "out_b", "type": "write_dataset", "params": {"directory": str(tmp_path),
                                                                         "subfolder": "b"}}]
    raw["edges"] += [{"from": "chunks.clip", "to": "seq_b.hr"}, {"from": "grain.clip", "to": "seq_b.lr"},
                     {"from": "seq_b.sequences", "to": "out_b.sequences"}]
    run(raw, emit=quiet)
    a, b = manifest(tmp_path / "a"), manifest(tmp_path / "b")
    assert len(a) == len(b) == 12 and [e["show"] for e in b] == list(range(1, 13))
    assert all("pick" not in e["settings"] and "grain" in e["settings"] for e in b)
    assert all("seq_b" not in e["settings"] for e in a)


@pytest.mark.parametrize("path", PRESETS, ids=os.path.basename)
def test_graph_presets_run(tmp_path, path):
    with open(path, encoding="utf-8") as f:
        raw = copy.deepcopy(json.load(f))
    for item in raw["nodes"]:
        params = item.setdefault("params", {})
        if item["type"] == "load_video":
            params["path"] = FIXTURE
        if item["type"] == "write_dataset":
            params["directory"] = str(tmp_path)
        if item["type"] == "chunk":
            params.update(strategy="frames", frames_per_chunk=24)
        if item["type"] in CLEAN_UP and importlib.util.find_spec(CLEAN_UP[item["type"]]) is None:
            item["enabled"] = False
    result = run(raw, emit=quiet)
    assert result.get("sequences", 0) > 0
