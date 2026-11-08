"""
Core tests for vk_dataset: graph checks, seeded sampling, chunking, the codec
round trip and the writer. Run from the repo root:

    python -m pytest tests/vk_dataset
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "include"))

import vk_dataset  # noqa: E402
from vk_dataset import Graph, GraphError, run  # noqa: E402
from vk_dataset.params import rng_for, sample  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "two_scenes.mkv")


def job(out_dir, seed=7, **overrides):
    nodes = {
        "src": {"type": "load_video", "params": {"path": FIXTURE}},
        "chunks": {"type": "chunk", "params": {"strategy": "scene", "min_duration": 0.5}},
        "down": {"type": "resample", "params": {"size": "scale", "scale": 0.5}},
        "enc": {"type": "codec", "params": {}},
        "seq": {"type": "make_sequences", "params": {"length": 5, "max_per_chunk": 2}},
        "out": {"type": "write_dataset", "params": {"directory": str(out_dir)}},
    }
    for node_id, params in overrides.items():
        nodes[node_id]["params"].update(params)
    return {
        "version": 1, "seed": seed,
        "nodes": [{"id": k, **v} for k, v in nodes.items()],
        "edges": [
            {"from": "src.clip", "to": "chunks.clip"},
            {"from": "chunks.clip", "to": "down.clip"},
            {"from": "down.clip", "to": "enc.clip"},
            {"from": "chunks.clip", "to": "seq.hr"},
            {"from": "enc.clip", "to": "seq.lr"},
            {"from": "seq.sequences", "to": "out.sequences"},
        ],
    }


def quiet(_):
    pass


def manifest(out_dir):
    with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)["sequences"]


def test_node_specs_are_json():
    specs = vk_dataset.node_specs()
    json.dumps(specs)
    assert {"load_video", "chunk", "resample", "codec", "make_sequences", "write_dataset"} <= {s["type"] for s in specs}


def test_rejects_mismatched_ports(tmp_path):
    raw = job(tmp_path)
    raw["edges"][4] = {"from": "seq.sequences", "to": "seq.lr"}
    with pytest.raises(GraphError):
        Graph(raw)


def test_rejects_cycles(tmp_path):
    raw = job(tmp_path)
    raw["edges"][1] = {"from": "enc.clip", "to": "down.clip"}
    with pytest.raises(GraphError, match="cycle"):
        Graph(raw)


def test_rejects_missing_input(tmp_path):
    raw = job(tmp_path)
    del raw["edges"][3]
    with pytest.raises(GraphError, match="needs an input"):
        Graph(raw)


def test_chunk_scope(tmp_path):
    graph = Graph(job(tmp_path))
    assert not graph.nodes["src"].chunk_scope
    assert all(graph.nodes[n].chunk_scope for n in ("chunks", "down", "enc", "seq", "out"))


def test_sampling_is_seeded():
    a = [sample({"uniform": [0, 1]}, rng_for(1, 0, n, "x")) for n in range(5)]
    b = [sample({"uniform": [0, 1]}, rng_for(1, 0, n, "x")) for n in range(5)]
    c = [sample({"uniform": [0, 1]}, rng_for(2, 0, n, "x")) for n in range(5)]
    assert a == b and a != c
    assert sample({"int": [3, 3]}, rng_for(0)) == 3
    assert sample({"choice": ["a", "b"], "weights": [1, 0]}, rng_for(0)) == "a"


def test_scene_split_and_pairs(tmp_path):
    result = run(job(tmp_path), emit=quiet)
    entries = manifest(tmp_path)
    assert result["sequences"] == 4 == len(entries)
    # The fixture cuts from testsrc2 to mandelbrot at 3 s, frame 72.
    assert sorted({e["chunk"] for e in entries}) == [0, 1]
    assert entries[2]["frames"]["hr"][0] == 72
    import cv2
    hr = cv2.imread(os.path.join(tmp_path, "HR", "show00001_Frame00001.png"))
    lr = cv2.imread(os.path.join(tmp_path, "LR", "show00001_Frame00005.png"))
    assert hr.shape == (240, 320, 3) and lr.shape == (120, 160, 3)


def test_same_seed_same_settings(tmp_path):
    run(job(tmp_path / "a", seed=3), emit=quiet)
    run(job(tmp_path / "b", seed=3), emit=quiet)
    run(job(tmp_path / "c", seed=4), emit=quiet)
    settings = lambda d: [e["settings"] for e in manifest(d)]  # noqa: E731
    assert settings(tmp_path / "a") == settings(tmp_path / "b")
    assert settings(tmp_path / "a") != settings(tmp_path / "c")


def test_dropdown_picks_from_the_ticked_options(tmp_path):
    pick = {"choice": ["xvid", "web_h265"], "weights": [1, 3]}
    run(job(tmp_path, enc={"profile": pick}), emit=quiet)
    drawn = {e["settings"]["enc"]["params"]["profile"] for e in manifest(tmp_path)}
    assert drawn and drawn <= {"xvid", "web_h265"}


def test_dropdown_pick_must_list_its_own_options(tmp_path):
    with pytest.raises(GraphError, match="mpeg4 is not one of"):
        Graph(job(tmp_path, enc={"profile": {"choice": ["xvid", "mpeg4"]}}))
    with pytest.raises(GraphError, match="one weight per option"):
        Graph(job(tmp_path, enc={"profile": {"choice": ["xvid"], "weights": [1, 2]}}))


def test_numbering_continues(tmp_path):
    run(job(tmp_path), emit=quiet)
    run(job(tmp_path), emit=quiet)
    shows = [e["show"] for e in manifest(tmp_path)]
    assert shows == list(range(1, 9))


def test_probability_zero_bypasses(tmp_path):
    run(job(tmp_path, enc={"probability": 0.0}, down={"probability": 0.0}), emit=quiet)
    import cv2
    hr = cv2.imread(os.path.join(tmp_path, "HR", "show00001_Frame00001.png"))
    lr = cv2.imread(os.path.join(tmp_path, "LR", "show00001_Frame00001.png"))
    assert hr.shape == lr.shape and (hr == lr).all()
    assert manifest(tmp_path)[0]["settings"].get("enc") is None


def test_center_frame_and_fixed_chunks(tmp_path):
    run(job(tmp_path, chunks={"strategy": "frames", "frames_per_chunk": 9},
            seq={"length": 9, "hr_frames": "center", "max_per_chunk": 0}), emit=quiet)
    entries = manifest(tmp_path)
    assert len(entries) == 144 // 9
    assert all(len(e["frames"]["hr"]) == 1 and e["frames"]["hr"][0] == e["frames"]["lr"][4] for e in entries)


@pytest.mark.parametrize("profile", ["web_h264", "dvd_mpeg2", "xvid"])
def test_codec_round_trip_keeps_length(profile):
    import vapoursynth as vs
    from vk_dataset.nodes.codec import profile_args
    from vk_dataset.media import round_trip
    clip = vs.core.std.BlankClip(width=64, height=48, format=vs.YUV420P8, length=17, fpsnum=24, fpsden=1)
    args, _ = profile_args(profile, rng_for(0), 0.5, 64, 48, 24, True)
    args += ["-pix_fmt", "yuv420p"]
    out = os.path.join(os.environ.get("TMP", "."), "vk_dataset_test.mkv")
    try:
        decoded = round_trip(clip, out, args)
        assert decoded.num_frames == 17 and decoded.format.id == clip.format.id
        decoded.get_frame(16)
        del decoded
    finally:
        import gc
        gc.collect()
        if os.path.exists(out):
            os.remove(out)
