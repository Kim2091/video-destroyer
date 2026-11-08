"""
Tests for the dataset preview (vk_dataset/preview.py): a previewed frame is
the frame a run writes for the same seed, chunk and frame, its settings are
the manifest's, sequences come with their crops and rejections, writers
never write, and a seed override draws anew.
"""

import json
import os
import struct
import subprocess
import sys

import pytest

INCLUDE = os.path.join(os.path.dirname(__file__), "..", "..", "include")
sys.path.insert(0, INCLUDE)

import vk_dataset  # noqa: E402,F401
from vk_dataset import run  # noqa: E402
from vk_dataset.preview import Preview, PreviewError  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "two_scenes.mkv")


def job(out_dir, seed=7, reject=None, **overrides):
    nodes = {
        "src": {"type": "load_video", "params": {"path": FIXTURE}},
        "chunks": {"type": "chunk", "params": {"strategy": "scene", "min_duration": 0.5}},
        "down": {"type": "resample", "params": {"size": "scale", "scale": 0.5}},
        "enc": {"type": "codec", "params": {}},
        "seq": {"type": "make_sequences", "params": {"length": 5, "max_per_chunk": 2}},
        "out": {"type": "write_dataset", "params": {"directory": str(out_dir)}},
    }
    edges = [
        {"from": "src.clip", "to": "chunks.clip"},
        {"from": "chunks.clip", "to": "down.clip"},
        {"from": "down.clip", "to": "enc.clip"},
        {"from": "chunks.clip", "to": "seq.hr"},
        {"from": "enc.clip", "to": "seq.lr"},
        {"from": "seq.sequences", "to": "out.sequences"},
    ]
    if reject:
        nodes["check"] = {"type": "reject_incomplete", "params": reject}
        edges[-1] = {"from": "seq.sequences", "to": "check.sequences"}
        edges.append({"from": "check.sequences", "to": "out.sequences"})
    for node_id, params in overrides.items():
        nodes[node_id]["params"].update(params)
    return {"version": 1, "seed": seed, "nodes": [{"id": k, **v} for k, v in nodes.items()], "edges": edges}


def manifest(out_dir):
    with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)["sequences"]


def read_rgb(path):
    import cv2
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


@pytest.fixture
def preview(tmp_path):
    work = tmp_path / "preview_work"
    work.mkdir()
    p = Preview(str(work))
    yield p
    p.close()


def test_frames_and_settings_match_a_run(tmp_path, preview):
    out = tmp_path / "run"
    run(job(out), emit=lambda _: None)
    entries = manifest(out)
    preview.load(job(tmp_path / "never"))

    chunks = preview.chunks(0)
    assert chunks["chunked"] and chunks["chunks"] == [[0, 72], [72, 144]]

    for show, entry in enumerate(entries, start=1):
        chunk = entry["chunk"]
        start = entry["frames"]["lr"][0] - chunks["chunks"][chunk][0]
        for place in range(5):
            header, (image,) = preview.frame("enc", "clip", 0, chunk, start + place)
            assert header["width"] == 160 and header["height"] == 120
            lr = read_rgb(os.path.join(out, "LR", f"show{show:05d}_Frame{place + 1:05d}.png"))
            assert (image == lr).all()
        header, (image,) = preview.frame("chunks", "clip", 0, chunk, start)
        assert (image == read_rgb(os.path.join(out, "HR", f"show{show:05d}_Frame00001.png"))).all()

        header, _ = preview.frame("seq", "sequences", 0, chunk, 0)
        expected = {k: v for k, v in entry["settings"].items() if k != "out"}
        assert header["settings"] == expected


def test_writers_never_write(tmp_path, preview):
    never, bad = tmp_path / "never", tmp_path / "bad"
    preview.load(job(never, reject={"length": 6}, out={"bad_directory": str(bad)}))
    preview.frame("check", "sequences", 0, 1, 3, compare={"node": "enc", "port": "clip"})
    assert not never.exists() and not bad.exists()
    with pytest.raises(PreviewError, match="no output"):
        preview.frame("out", "sequences", 0, 0, 0)


def test_sequences_and_rejections(tmp_path, preview):
    preview.load(job(tmp_path / "never", reject={"length": 6}))
    header, (image,) = preview.frame("seq", "sequences", 0, 0, 2, view="lr")
    assert header["kind"] == "sequences" and header["view"] == "lr" and image.shape == (120, 160, 3)
    sequences = header["sequences"]
    assert sequences["clips"]["hr"] == {"width": 320, "height": 240, "frames": 72}
    assert [s["frames"]["hr"] for s in sequences["items"]] == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    assert sequences["rejected"] == []

    header, _ = preview.frame("check", "sequences", 0, 0, 0)
    assert header["sequences"]["items"] == []
    assert [s["info"]["rejected"] for s in header["sequences"]["rejected"]] == ["incomplete", "incomplete"]
    assert header["settings"]["check"]["params"] == {"rejected": 2}


def test_tiles_carry_crops(tmp_path, preview):
    raw = job(tmp_path / "never")
    raw["nodes"].append({"id": "tiles", "type": "tile", "params": {"tile_width": 128, "tile_height": 96}})
    raw["edges"][-1] = {"from": "seq.sequences", "to": "tiles.sequences"}
    raw["edges"].append({"from": "tiles.sequences", "to": "out.sequences"})
    preview.load(raw)
    header, _ = preview.frame("tiles", "sequences", 0, 0, 0)
    items = header["sequences"]["items"]
    assert len(items) == 2 * 4
    assert all(i["crops"]["hr"][2:] == [128, 96] and i["crops"]["lr"][2:] == [64, 48] for i in items)


def test_reroll_draws_anew_and_returns(tmp_path, preview):
    preview.load(job(tmp_path / "never"))
    base, _ = preview.frame("enc", "clip", 0, 0, 0)
    assert base["seed"] == 7
    rerolls = [preview.frame("enc", "clip", 0, 0, 0, seed=s)[0]["settings"] for s in (100, 101, 102)]
    assert any(r != base["settings"] for r in rerolls)
    again, _ = preview.frame("enc", "clip", 0, 0, 0)
    assert again["settings"] == base["settings"]


def test_chunk_cache_and_temp_files(tmp_path, preview):
    preview.load(job(tmp_path / "never"))
    first, _ = preview.frame("enc", "clip", 0, 0, 0)
    scrub, _ = preview.frame("enc", "clip", 0, 0, 10)
    assert not first["cached"] and scrub["cached"]
    assert len(os.listdir(preview.work_dir)) == 1
    preview.frame("enc", "clip", 0, 1, 0)
    # The first chunk's encode is gone with its chunk.
    assert len(os.listdir(preview.work_dir)) == 1


def test_edits_keep_unchanged_outputs(tmp_path, preview):
    preview.load(job(tmp_path / "never"))
    preview.frame("seq", "sequences", 0, 0, 0)
    encoded = preview.chunk.values["enc"]
    loaded = preview.load(job(tmp_path / "never", seq={"max_per_chunk": 1}))
    assert loaded["reused"] == 4
    header, _ = preview.frame("seq", "sequences", 0, 0, 0)
    assert header["cached"] and preview.chunk.values["enc"] is encoded
    assert len(header["sequences"]["items"]) == 1

    preview.load(job(tmp_path / "never", down={"scale": 0.25}))
    header, (image,) = preview.frame("enc", "clip", 0, 0, 0)
    assert image.shape == (60, 80, 3) and "enc" in header["settings"]


def test_per_video_compare_shows_the_chunk(tmp_path, preview):
    preview.load(job(tmp_path / "never"))
    header, (lr, hr) = preview.frame("enc", "clip", 0, 1, 4, compare={"node": "src", "port": "clip"})
    assert header["compare"]["scope"] == "video" and header["compare"]["offset"] == 72
    assert header["compare"]["n"] == 4 and hr.shape == (240, 320, 3) and lr.shape == (120, 160, 3)
    _, (chunk_hr,) = preview.frame("chunks", "clip", 0, 1, 4)
    assert (hr == chunk_hr).all()


def test_errors_name_the_problem(tmp_path, preview):
    with pytest.raises(PreviewError, match="no job"):
        preview.frame("enc", "clip", 0, 0, 0)
    preview.load(job(tmp_path / "never"))
    with pytest.raises(PreviewError, match="no chunk 5"):
        preview.frame("enc", "clip", 0, 5, 0)
    with pytest.raises(PreviewError, match="no video 3"):
        preview.frame("enc", "clip", 3, 0, 0)
    with pytest.raises(PreviewError, match="no node"):
        preview.frame("nope", "clip", 0, 0, 0)


def read_replies(stream):
    replies = []
    while True:
        size = stream.read(4)
        if len(size) < 4:
            return replies
        header = json.loads(stream.read(struct.unpack(">I", size)[0]))
        payload = stream.read(header.get("bytes", 0)) if header.get("bytes") else b""
        replies.append((header, payload))


def test_server_protocol(tmp_path):
    bootstrap = ("import sys; sys.path.insert(0, sys.argv.pop(1)); "
                 "from vk_dataset.__main__ import main; sys.exit(main())")
    commands = [
        {"cmd": "load", "seq": 1, "job": job(tmp_path / "never")},
        {"cmd": "chunks", "seq": 5, "video": 0},
        {"cmd": "frame", "seq": 2, "node": "enc", "port": "clip", "video": 0, "chunk": 0, "frame": 0},
        {"cmd": "frame", "seq": 3, "node": "enc", "port": "clip", "video": 0, "chunk": 0, "frame": 1,
         "compare": {"node": "chunks", "port": "clip"}},
        {"cmd": "ping", "seq": 7},
        {"cmd": "frame", "seq": 4, "node": "enc", "port": "clip", "video": 0, "chunk": 9, "frame": 0},
        {"cmd": "close", "seq": 6},
    ]
    stdin = "".join(json.dumps(c) + "\n" for c in commands).encode()
    done = subprocess.run([sys.executable, "-u", "-c", bootstrap, os.path.abspath(INCLUDE), "preview"],
                          input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    assert done.returncode == 0, done.stderr.decode(errors="replace")
    import io
    replies = read_replies(io.BytesIO(done.stdout))
    by_seq = {h["seq"]: (h, p) for h, p in replies}
    assert by_seq[1][0]["type"] == "loaded" and by_seq[1][0]["videos"] == [{"index": 0, "name": "two_scenes"}]
    # The scene split takes long enough for both frame requests to arrive behind it.
    assert by_seq[2][0]["type"] == "superseded"
    header, payload = by_seq[3]
    assert header["type"] == "frame" and header["compare"]["width"] == 320
    assert len(payload) == header["bytes"] == 160 * 120 * 3 + 320 * 240 * 3
    assert by_seq[4][0]["type"] == "error" and "no chunk 9" in by_seq[4][0]["error"]
    assert by_seq[5][0]["chunks"] == [[0, 72], [72, 144]]
    assert by_seq[6][0]["type"] == by_seq[7][0]["type"] == "ok"


def test_previews_a_job_without_a_writer(tmp_path, preview):
    raw = job(tmp_path / "never")
    raw["nodes"] = [n for n in raw["nodes"] if n["id"] != "out"]
    raw["edges"] = [e for e in raw["edges"] if not e["to"].startswith("out.")]
    preview.load(raw)
    header, _ = preview.frame("seq", "sequences", 0, 0, 0)
    assert header["clips"] == ["hr", "lr"] and header["view"] == "hr"
