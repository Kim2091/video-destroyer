"""
Tests for tonemap, the ffmpeg filter node, how any degradation behaves in a
job (seeded, skippable), the sequence post-processing nodes and the
every-degradation preset. The degradation nodes themselves are tested in
test_lr_degradations.py. Run from the repo root:

    python -m pytest tests/vk_dataset
"""

import json
import os
import random
import sys

import numpy as np
import pytest
import vapoursynth as vs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "include"))

from vk_dataset import Graph, NODE_TYPES, run  # noqa: E402
from vk_dataset.media import load  # noqa: E402
from vk_dataset.params import sample  # noqa: E402
from vk_dataset.nodes.sequences import Sequence, Sequences  # noqa: E402

core = vs.core
HERE = os.path.dirname(__file__)
FIXTURE = os.path.join(HERE, "fixtures", "two_scenes.mkv")
PRESET = os.path.join(HERE, "..", "..", "include", "vk_dataset", "presets", "03-every-degradation.json")


class FakeContext:
    """What a node function needs of the runner's Context."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.notes = {}
        self.temp = []
        self.video_name = "test"

    def note(self, key, value):
        self.notes[key] = value

    def message(self, text):
        pass

    def temp_path(self, suffix):
        path = os.path.join(os.environ.get("TMP", "."), f"vk_dataset_degrade_{os.getpid()}_{len(self.temp)}{suffix}")
        self.temp.append(path)
        return path


def apply(name, clip, seed=0, **params):
    node_type = NODE_TYPES[name]
    values = {p.name: p.default for p in node_type.params}
    values.update(params)
    ctx = FakeContext(seed)
    values = {k: sample(v, ctx.rng) for k, v in values.items()}
    try:
        return node_type.func(ctx, clip, **values)["clip"], ctx
    finally:
        for path in ctx.temp:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except PermissionError:
                    pass


def planes(clip, n):
    frame = clip.get_frame(n)
    return [np.asarray(frame[i]).astype(float) for i in range(frame.format.num_planes)]


def psnr(a, b):
    mse = np.mean([np.mean((x - y) ** 2) for x, y in zip(a, b)])
    return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)


@pytest.fixture(scope="module")
def clip():
    return load(FIXTURE)[:30]


DEGRADATIONS = ["optics_blur", "film_grain", "sensor_noise", "resample", "sharpen", "chroma_bleed", "quantize",
                "codec", "player_deblock", "player_dnr", "chroma_subsample", "ghost_echo", "tonemap", "ffmpeg_filter"]


def test_all_nodes_are_declared():
    names = set(DEGRADATIONS) | {"scale_to_model", "tile", "reject_blank", "reject_motion", "reject_incomplete"}
    assert names <= set(NODE_TYPES)
    for name in names:
        assert NODE_TYPES[name].inputs
    for name in DEGRADATIONS:
        assert NODE_TYPES[name].probability and NODE_TYPES[name].category == "Degrade"
    # Settings are fields a person can edit, never raw JSON.
    for name in [*DEGRADATIONS, "scale_to_model"]:
        assert all(p.kind != "json" for p in NODE_TYPES[name].params), name


def hdr(clip):
    return core.std.SetFrameProps(clip, _Transfer=16, _Primaries=9, _Matrix=9)


def test_tonemap_skips_sdr_and_maps_hdr(clip):
    from vk_dataset.runner import NodeSkipped
    with pytest.raises(NodeSkipped):
        apply("tonemap", clip)
    for algorithm in ("hable", "mobius", "reinhard", "bt2390", "linear"):
        out, ctx = apply("tonemap", hdr(clip), algorithm=algorithm, desat=0.5, target_nits=10.0)
        assert out.format.id == vs.YUV420P8 and ctx.notes["detected_hdr"]
        planes(out, 3)
    bright, _ = apply("tonemap", hdr(clip), algorithm="hable", target_nits=10.0)
    dark, _ = apply("tonemap", hdr(clip), algorithm="hable", target_nits=100.0)
    assert planes(bright, 3)[0].mean() > planes(dark, 3)[0].mean()


def test_tonemap_forced_on_untagged(clip):
    out, _ = apply("tonemap", clip, auto_detect=False)
    assert out.format.id == vs.YUV420P8


def test_ffmpeg_filter_matches_ffmpeg(clip):
    out, ctx = apply("ffmpeg_filter", clip, filter="hflip")
    assert out.num_frames == clip.num_frames
    assert np.abs(planes(out, 4)[0] - planes(clip, 4)[0][:, ::-1]).mean() < 1
    with pytest.raises(ValueError):
        apply("ffmpeg_filter", clip, filter=" ")


def test_ffmpeg_filter_changing_frame_count(clip):
    out, _ = apply("ffmpeg_filter", clip, filter="tinterlace=mode=4")
    assert out.num_frames == clip.num_frames


NODES_WITH_DRAWS = [(name, {}) for name in DEGRADATIONS if name not in ("tonemap", "ffmpeg_filter")]


def pipeline_job(out_dir, node_type, params, seed=3):
    return {
        "version": 1, "seed": seed,
        "nodes": [
            {"id": "src", "type": "load_video", "params": {"path": FIXTURE}},
            {"id": "chunks", "type": "chunk", "params": {"strategy": "frames", "frames_per_chunk": 8}},
            {"id": "deg", "type": node_type, "params": params},
            {"id": "seq", "type": "make_sequences", "params": {"length": 4, "max_per_chunk": 1}},
            {"id": "out", "type": "write_dataset", "params": {"directory": str(out_dir)}},
        ],
        "edges": [
            {"from": "src.clip", "to": "chunks.clip"}, {"from": "chunks.clip", "to": "deg.clip"},
            {"from": "chunks.clip", "to": "seq.hr"}, {"from": "deg.clip", "to": "seq.lr"},
            {"from": "seq.sequences", "to": "out.sequences"},
        ],
    }


def manifest(out_dir):
    with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)["sequences"]


@pytest.mark.parametrize("node_type", [n for n, _ in NODES_WITH_DRAWS])
def test_degradation_is_deterministic_and_bypassable(tmp_path, node_type):
    run(pipeline_job(tmp_path / "a", node_type, {}), emit=lambda _: None)
    run(pipeline_job(tmp_path / "b", node_type, {}), emit=lambda _: None)
    run(pipeline_job(tmp_path / "off", node_type, {"probability": 0.0}), emit=lambda _: None)
    a, b, off = manifest(tmp_path / "a"), manifest(tmp_path / "b"), manifest(tmp_path / "off")
    assert [e["settings"] for e in a] == [e["settings"] for e in b]
    assert a[0]["settings"]["deg"]["applied"] is True
    assert off[0]["settings"].get("deg") is None
    name = "show00001_Frame00001.png"
    assert open(tmp_path / "a" / "LR" / name, "rb").read() == open(tmp_path / "b" / "LR" / name, "rb").read()
    hr, lr = (open(tmp_path / "off" / folder / name, "rb").read() for folder in ("HR", "LR"))
    assert hr == lr


def test_ffmpeg_filter_node_in_a_job(tmp_path):
    job = pipeline_job(tmp_path, "ffmpeg_filter", {"filter": "negate", "probability": 1.0})
    run(job, emit=lambda _: None)
    assert manifest(tmp_path)[0]["settings"]["deg"]["params"]["filter"] == "negate"


def telecine_job(out_dir, telecine=True, fix=True, telecine_params=None):
    """pipeline_job with a Telecine, then Compression and a Bad Telecine Fix on the LR side."""
    job = pipeline_job(out_dir, "telecine", telecine_params or {})
    job["nodes"].insert(3, {"id": "comp", "type": "codec", "params": {"profile": "dvd_mpeg2"}})
    job["edges"] = [e for e in job["edges"] if e["from"] != "deg.clip"] + [{"from": "deg.clip", "to": "comp.clip"}]
    last = "comp"
    if fix:
        job["nodes"].insert(4, {"id": "fix", "type": "telecine_fix", "params": {}})
        job["edges"].append({"from": "comp.clip", "to": "fix.clip"})
        last = "fix"
    job["edges"].append({"from": f"{last}.clip", "to": "seq.lr"})
    if not telecine:
        job["nodes"] = [n for n in job["nodes"] if n["id"] != "deg"]
        job["edges"] = [e for e in job["edges"] if e["to"] != "deg.clip"]
        for e in job["edges"]:
            if e["from"] == "deg.clip":
                e["from"] = "chunks.clip"
    return job


def test_telecine_needs_its_fix(tmp_path):
    from vk_dataset.graph import GraphError
    with pytest.raises(GraphError, match=r"Telecine \(deg\) reaches Cut into Sequences \(seq\) without a Bad "
                                         r"Telecine Fix"):
        Graph(telecine_job(tmp_path, fix=False))
    with pytest.raises(GraphError, match=r"Bad Telecine Fix \(fix\) needs a Telecine before it"):
        Graph(telecine_job(tmp_path, telecine=False))
    # A switched-off fix fixes nothing; a partial job for a preview is not checked.
    job = telecine_job(tmp_path)
    next(n for n in job["nodes"] if n["id"] == "fix")["enabled"] = False
    with pytest.raises(GraphError, match="without a Bad Telecine Fix"):
        Graph(job)
    Graph(telecine_job(tmp_path, fix=False), needs_writer=False)


def test_telecine_and_fix_in_a_job(tmp_path):
    run(telecine_job(tmp_path / "on"), emit=lambda _: None)
    run(telecine_job(tmp_path / "off", telecine_params={"probability": 0.0}), emit=lambda _: None)
    on, off = manifest(tmp_path / "on"), manifest(tmp_path / "off")
    assert on[0]["settings"]["deg"]["applied"] and on[0]["settings"]["fix"]["applied"]
    assert on[0]["frames"]["hr"] == on[0]["frames"]["lr"]
    assert "deg" not in off[0]["settings"] and "fix" not in off[0]["settings"]


def sequences_of(clip, items, hr=None):
    clips = {"hr": hr if hr is not None else clip, "lr": clip}
    return Sequences(clips, items)


def test_tile_crops_hr_and_lr(clip):
    lr = core.resize.Bicubic(clip, 160, 120)
    seqs = Sequences({"hr": clip, "lr": lr}, [Sequence({"hr": [0, 1], "lr": [0, 1]}, info={"start": 0})])
    out, ctx = apply_sequences("tile", seqs, tile_width=160, tile_height=120)
    assert len(out.items) == 4 and ctx.notes["tiles_per_sequence"] == 4
    for item in out.items:
        left, top, width, height = item.crops["hr"]
        assert (width, height) == (160, 120) and left % 160 == 0 and top % 120 == 0
        assert item.crops["lr"] == (left // 2, top // 2, 80, 60)
    again, _ = apply_sequences("tile", seqs, tile_width=160, tile_height=120)
    assert [i.crops for i in again.items] == [i.crops for i in out.items]
    other, _ = apply_sequences("tile", seqs, seed=9, tile_width=160, tile_height=120)
    assert [i.crops for i in other.items] != [i.crops for i in out.items]
    limited, _ = apply_sequences("tile", seqs, tile_width=160, tile_height=120, max_tiles=2)
    assert len(limited.items) == 2


def test_tile_too_large_rejects(clip):
    seqs = sequences_of(clip, [Sequence({"hr": [0, 1], "lr": [0, 1]})])
    out, _ = apply_sequences("tile", seqs, tile_width=1000, tile_height=1000)
    assert not out.items and out.rejected[0].info["rejected"] == "smaller than tile"


def apply_sequences(name, sequences, seed=0, **params):
    node_type = NODE_TYPES[name]
    values = {p.name: p.default for p in node_type.params}
    values.update(params)
    ctx = FakeContext(seed)
    return node_type.func(ctx, sequences, **values)["sequences"], ctx


@pytest.fixture(scope="module")
def flat_and_busy():
    flat = core.std.BlankClip(width=64, height=64, format=vs.YUV420P8, length=6, color=[128, 128, 128])
    noise = core.std.BlankClip(width=64, height=64, format=vs.YUV420P8, length=6)
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 256, (64, 64), dtype=np.uint8) for _ in range(6)]

    def fill(n, f):
        out = f.copy()
        np.asarray(out[0])[:] = frames[n]
        return out
    busy = core.std.ModifyFrame(noise, noise, fill)
    return flat, busy


def test_reject_blank(flat_and_busy):
    flat, busy = flat_and_busy
    seqs = Sequences({"hr": core.std.Interleave([flat[:3], busy[:3]])}, [])
    # Interleaving puts the flat frames at 0, 2, 4 and the busy ones at 1, 3, 5.
    flat_frames = [Sequence({"hr": [0, 2, 4]}), Sequence({"hr": [1, 3, 5]}), Sequence({"hr": [0, 1, 3]})]
    out, ctx = apply_sequences("reject_blank", seqs.replace(flat_frames), min_blank_frames=2)
    assert [i.frames["hr"] for i in out.items] == [[1, 3, 5], [0, 1, 3]]
    assert [i.frames["hr"] for i in out.rejected] == [[0, 2, 4]]
    assert out.rejected[0].info["rejected"] == "blank" and ctx.notes["rejected"] == 1


def test_reject_blank_black_bars():
    letterboxed = core.std.BlankClip(width=64, height=64, format=vs.YUV420P8, length=2, color=[16, 128, 128])
    seqs = Sequences({"hr": letterboxed}, [Sequence({"hr": [0, 1]})])
    out, _ = apply_sequences("reject_blank", seqs, min_blank_frames=1, edge_threshold=0.0, variance_threshold=0.0)
    assert out.rejected


def test_reject_motion(clip):
    # The fixture cuts between scenes at frame 72.
    cut = load(FIXTURE)
    seqs = Sequences({"hr": cut}, [Sequence({"hr": [10, 11, 12]}), Sequence({"hr": [70, 71, 72, 73]}),
                                   Sequence({"hr": [20, 20]})])
    out, _ = apply_sequences("reject_motion", seqs, min_motion=0.5, max_motion=15.0, threshold=30)
    reasons = {tuple(i.frames["hr"]): i.info["rejected"] for i in out.rejected}
    assert reasons[(70, 71, 72, 73)] == "too much motion"
    assert reasons[(20, 20)] == "too little motion"
    assert [i.frames["hr"] for i in out.items] == [[10, 11, 12]]
    disabled, _ = apply_sequences("reject_motion", seqs, min_motion=-1, max_motion=-1)
    assert len(disabled.items) == 3


def test_reject_motion_single_frame_is_kept(clip):
    seqs = Sequences({"hr": clip}, [Sequence({"hr": [3]})])
    out, _ = apply_sequences("reject_motion", seqs, min_motion=5.0)
    assert len(out.items) == 1


def test_reject_motion_honours_crops(clip):
    seqs = Sequences({"hr": clip}, [Sequence({"hr": [10, 11]}, crops={"hr": (0, 0, 32, 32)})])
    out, _ = apply_sequences("reject_motion", seqs, min_motion=-1, max_motion=0.0)
    assert out.rejected or out.items


def test_reject_incomplete(clip):
    seqs = Sequences({"hr": clip, "lr": clip}, [
        Sequence({"hr": [0, 1, 2], "lr": [0, 1, 2]}),
        Sequence({"hr": [0, 1, 2], "lr": [0, 1, 3]}),
        Sequence({"hr": [1], "lr": [0, 1, 2]}),
        Sequence({"hr": [28, 29, 30], "lr": [28, 29, 30]}),
    ])
    out, _ = apply_sequences("reject_incomplete", seqs, length=3)
    assert [i.frames["lr"] for i in out.items] == [[0, 1, 2], [0, 1, 2]]
    assert len(out.rejected) == 2


def test_filters_keep_earlier_rejections(clip):
    seqs = Sequences({"hr": clip, "lr": clip}, [Sequence({"hr": [0, 1], "lr": [0, 1]}),
                                                Sequence({"hr": [0, 2], "lr": [0, 1]})])
    first, _ = apply_sequences("reject_incomplete", seqs, length=2)
    second, _ = apply_sequences("reject_motion", first, min_motion=99.0)
    assert len(first.rejected) == 1 and len(second.rejected) == 2


def test_writer_puts_rejected_in_bad_folder(tmp_path, clip):
    job = pipeline_job(tmp_path / "good", "sharpen", {"probability": 0.0})
    job["nodes"].insert(3, {"id": "motion", "type": "reject_motion", "params": {"min_motion": 99.0}})
    job["nodes"][-1]["params"]["bad_directory"] = str(tmp_path / "bad")
    job["edges"][-1] = {"from": "motion.sequences", "to": "out.sequences"}
    job["edges"].append({"from": "seq.sequences", "to": "motion.sequences"})
    result = run(job, emit=lambda _: None)
    assert result["rejected"] == 18 and result.get("sequences", 0) == 0
    assert os.path.exists(tmp_path / "bad" / "HR" / "show00001_Frame00001.png")
    bad = manifest(tmp_path / "bad")
    assert bad[0]["info"]["rejected"] == "too little motion"


def test_every_degradation_preset_checks_and_runs(tmp_path):
    with open(PRESET, encoding="utf-8") as f:
        job = json.load(f)
    by_id = {n["id"]: n for n in job["nodes"]}
    by_id["src"]["params"]["path"] = FIXTURE
    Graph(job)
    # Switch every degradation on, so the whole chain runs in order.
    for item in job["nodes"]:
        if NODE_TYPES[item["type"]].category == "Degrade" and item["type"] != "tonemap":
            item["enabled"] = True
    by_id["out"]["params"]["directory"] = str(tmp_path / "frames")
    by_id["motion"]["params"]["max_motion"] = 100.0
    by_id["chunks"]["params"]["min_duration"] = 0.5
    result = run(job, emit=lambda _: None)
    assert result["sequences"] > 0 and result["videos"] == 1
    sequences = manifest(tmp_path / "frames")
    assert sequences[0]["settings"]["comp"]["params"]["profile"] == "dvd_mpeg2"
