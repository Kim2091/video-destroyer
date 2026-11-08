"""
Pair mode tests: time and space alignment of an LR release to an HR one,
sequence picking, detail tiles and the alignment gates, and
VSR-Dataset-Creator's output contract (LR show<N>_Frame1..9, HR
show<N>_Frame5 only).

fixtures/pair_split_expected.json holds what the single pair node kept
before it became Pick Aligned Sequences, Cut into Tiles and Drop Misaligned,
for the same inputs and settings; the three nodes must keep exactly that.

The LR release is made from the fixture with ffmpeg: three frames late,
cropped off-centre, halved and re-encoded. vs_align runs on CUDA when
PyTorch sees a GPU and on the CPU otherwise; the CUDA-only check is skipped
without one. Set VK_DATASET_CONVNEXT_WEIGHTS to the ConvNeXt-S weights file
to run the dedup test.
"""

import json
import os
import subprocess
import sys

import numpy as np
import importlib.util

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "include"))

import vk_dataset  # noqa: E402
from vk_dataset import Graph, run  # noqa: E402
from vk_dataset.media import load  # noqa: E402
from vk_dataset.nodes import pair_align, pair_select  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "two_scenes.mkv")
PRESET = os.path.join(os.path.dirname(__file__), "..", "..", "include", "vk_dataset", "presets",
                      "06-two-releases-to-aligned-pairs.json")
SHIFT = 3
WEIGHTS = os.environ.get("VK_DATASET_CONVNEXT_WEIGHTS", "")


def cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture(scope="module")
def lr_video(tmp_path_factory):
    """The fixture as a worse release: SHIFT frames late, cropped 6/4/4/2 px, halved, x264 CRF 20."""
    path = str(tmp_path_factory.mktemp("pair") / "lr.mkv")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", FIXTURE, "-vf",
                    f"tpad=start={SHIFT}:start_mode=clone,trim=end_frame=144,setpts=PTS-STARTPTS,"
                    "crop=310:234:6:4,scale=160:120:flags=bicubic",
                    "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", path], check=True)
    return path


class StubContext:
    video_name = "pair"

    def message(self, text):
        pass

    def note(self, key, value):
        pass


def pair_job(out_dir, lr_path, dedup_weights=None, **overrides):
    with open(PRESET, encoding="utf-8") as f:
        raw = json.load(f)
    nodes = {n["id"]: n for n in raw["nodes"]}
    nodes["hr_src"]["params"]["path"] = FIXTURE
    nodes["lr_src"]["params"]["path"] = lr_path
    nodes["out"]["params"]["directory"] = str(out_dir)
    # The fixture is 320x240, so crops are smaller and may touch the edge.
    nodes["tiles"]["params"].update({"tile_width": 128, "tile_height": 128, "edge_margin": 0})
    for node_id, params in overrides.items():
        nodes[node_id]["params"].update(params)
    if dedup_weights:
        nodes["dedup"]["params"]["weights"] = dedup_weights
    else:
        raw["nodes"] = [n for n in raw["nodes"] if n["id"] != "dedup"]
        raw["edges"] = [e for e in raw["edges"] if not e["to"].startswith("dedup.")]
        for edge in raw["edges"]:
            if edge["from"] == "dedup.sequences":
                edge["from"] = "aligned.sequences"
    return raw


def manifest(out_dir):
    with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)["sequences"]


def shift_px(a, b):
    """How far, in pixels, b's picture sits from a's (phase correlation of grey images)."""
    import cv2
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(ga, gb, cv2.createHanningWindow(ga.shape[::-1], cv2.CV_32F))
    return float(np.hypot(dx, dy))


def test_nodes_are_registered_and_preset_loads():
    types = {s["type"] for s in vk_dataset.node_specs()}
    assert {"crop_rescale", "align_time", "align_space", "pair_sequences", "dedup_sequences"} <= types
    graph = Graph.load(PRESET)
    assert graph.chunker.id == "chunks"
    assert not graph.nodes["space"].chunk_scope and graph.nodes["seq"].chunk_scope


def test_crop_rescale_sizes():
    import vapoursynth as vs
    clip = vs.core.std.BlankClip(width=320, height=240, format=vs.YUV420P8, length=2)
    out = pair_align.crop_rescale(None, clip, left=6, right=4, top=4, bottom=3, rescale=0.5)["clip"]
    assert (out.width, out.height) == (155, 116)
    out.get_frame(0)


def test_remap_moves_frames():
    import vapoursynth as vs
    clip = pair_align.tag_frame_numbers(vs.core.std.BlankClip(width=16, height=16, format=vs.GRAY8, length=10))
    out = pair_align.remap(clip, 12, lambda n: n - 2)
    assert [f.props["LRFrame"] for f in out.frames()] == [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_timing_drift_and_windows():
    assert pair_select.timing_drift([10, 11, 12, 13, 14], 2, 0) == (0, 0)
    assert pair_select.timing_drift([10, 11, 12, 12, 13], 2, 0) == (2, 2)
    motion = np.zeros(40, dtype=np.float32)
    motion[20:28] = 1.0
    assert pair_select.pick_windows(motion, 41, 9, 4, 1)[0][0] == 20


def test_weights_cannot_push_a_flat_crop_past_the_threshold():
    complexity = np.zeros((64, 64), dtype=np.float32)
    complexity[:, 32:] = 0.01
    weights = np.ones_like(complexity)
    weights[:, :32] = 1000.0
    picked = pair_select.select_tiles(complexity, 32, 32, 2, 1, 0.005, 0, weights)
    assert picked and picked[0][0] == 32


EXPECTED = os.path.join(os.path.dirname(__file__), "fixtures", "pair_split_expected.json")


def split_inputs():
    """
    HR is the fixture; LR is it halved, its right half a pixel off, and two
    stretches of frames a frame late, so every gate drops something.
    """
    import vapoursynth as vs
    core = vs.core
    hr = load(FIXTURE)
    lr = core.resize.Bicubic(hr, 160, 120)
    off = core.resize.Bicubic(hr, 160, 120, src_left=2)
    lr = core.std.StackHorizontal([core.std.Crop(lr, right=80), core.std.Crop(off, left=80)])
    late = lambda n: n - 1 if 113 <= n < 118 or 133 <= n < 137 else n  # noqa: E731
    return hr, pair_align.remap(pair_align.tag_frame_numbers(lr), lr.num_frames, late)


class SplitContext(StubContext):
    def __init__(self):
        import random
        self.rng = random.Random(0)


def split_settings(old):
    """The settings of the single pair node, as the three nodes that replace it take them."""
    pick = {k: old[k] for k in ("length", "per_chunk", "margin")}
    tiles = {"pick": "detail", "tile_width": old["tile_size"], "tile_height": old["tile_size"],
             "max_tiles": old["crops_per_sequence"] + old["crop_align_candidates"],
             "min_detail": old["tile_threshold"], "edge_margin": old["crop_edge_margin"],
             "motion_bias": old["crop_motion_bias"], "reference": "hr"}
    drop = {k: old[k] for k in ("min_alignment_score", "min_crop_alignment_score", "max_drift", "drift_frames")}
    drop["crops_per_run"] = old["crops_per_sequence"]
    return pick, tiles, drop


def test_split_pair_nodes_keep_what_the_single_node_kept():
    from vk_dataset.nodes import post
    with open(EXPECTED, encoding="utf-8") as f:
        expected = json.load(f)
    hr, lr = split_inputs()
    rejected = set()
    for name, scenario in expected["scenarios"].items():
        pick, tiles, drop = split_settings(scenario["params"])
        kept = []
        for start, end in expected["chunks"]:
            ctx = SplitContext()
            sequences = pair_select.pair_sequences(ctx, hr[start:end], lr[start:end], **pick)["sequences"]
            sequences = post.tile(ctx, sequences, **tiles)["sequences"]
            sequences = pair_select.reject_misaligned(ctx, sequences, **drop)["sequences"]
            rejected |= {item.info["rejected"] for item in sequences.rejected}
            kept += [{"chunk": [start, end], "frames": item.frames,
                      "crops": {k: list(v) for k, v in item.crops.items()},
                      "align_score": item.info["align_score"],
                      "crop_align_score": item.info.get("crop_align_score")} for item in sequences.items]
        assert kept == scenario["sequences"], name
    # Every gate dropped something somewhere, so each one was compared.
    assert {"out of sync", "misaligned", "crop misaligned"} <= rejected


@pytest.mark.skipif(importlib.util.find_spec("vs_align") is None, reason="vs_align is not installed")
def test_align_time_recovers_the_offset(lr_video):
    hr, lr = load(FIXTURE), load(lr_video)
    matched = pair_align.align_time(StubContext(), hr, lr, radius=3, precision=1, backend="auto")["lr"]
    assert matched.num_frames == hr.num_frames
    offsets = np.array([f.props["LRFrame"] - n for n, f in enumerate(matched.frames(close=True))])
    # The last frames have no LR partner left and repeat LR's last frame.
    inner = offsets[: hr.num_frames - SHIFT]
    assert np.median(inner) == SHIFT
    assert np.mean(inner == SHIFT) >= 0.85


@pytest.mark.skipif(importlib.util.find_spec("vs_align") is None, reason="vs_align is not installed")
def test_pair_pipeline_writes_the_centre_frame_contract(tmp_path, lr_video):
    result = run(pair_job(tmp_path, lr_video), emit=lambda _: None)
    entries = manifest(tmp_path)
    assert result["sequences"] == len(entries) >= 2
    hr_files = sorted(os.listdir(tmp_path / "hr"))
    lr_files = sorted(os.listdir(tmp_path / "lr"))
    shows = sorted(e["show"] for e in entries)
    assert hr_files == sorted(f"show{s}_Frame5.png" for s in shows)
    assert lr_files == sorted(f"show{s}_Frame{i}.png" for s in shows for i in range(1, 10))

    import cv2
    for entry in entries:
        assert entry["frames"]["hr"] == [entry["frames"]["lr"][4]]
        # Every LR frame came from SHIFT frames later in the LR release.
        assert [s - f for s, f in zip(entry["info"]["lr_source_frames"], entry["frames"]["lr"])] == [SHIFT] * 9
        assert entry["info"]["align_score"] >= 0.70 and entry["info"]["crop_align_score"] >= 0.82
        show = entry["show"]
        hr = cv2.imread(str(tmp_path / "hr" / f"show{show}_Frame5.png"))
        lr = cv2.imread(str(tmp_path / "lr" / f"show{show}_Frame5.png"))
        assert hr.shape == (128, 128, 3) and lr.shape == (64, 64, 3)
        # The HR target shrunk to LR size sits on the LR centre frame.
        assert shift_px(cv2.resize(hr, (64, 64), interpolation=cv2.INTER_AREA), lr) < 0.5


@pytest.mark.skipif(not WEIGHTS or not os.path.isfile(WEIGHTS), reason="VK_DATASET_CONVNEXT_WEIGHTS is not set")
def test_dedup_drops_near_duplicates(tmp_path, lr_video):
    result = run(pair_job(tmp_path, lr_video, dedup_weights=WEIGHTS, dedup={"threshold": 1e9}), emit=lambda _: None)
    assert result["sequences"] == 1 and result["duplicates_dropped"] >= 1
    assert len(os.listdir(tmp_path / "lr")) == 9


@pytest.mark.skipif(not cuda_available(), reason="CUDA is not available")
def test_cuda_and_cpu_align_alike(lr_video):
    hr, lr = load(FIXTURE)[:-SHIFT], load(lr_video)[SHIFT:]
    frames = {}
    for backend in ("cuda", "cpu"):
        out = pair_align.align_space(None, hr, lr, scale=2, precision=3, wide_search=True, lq_input=True,
                                     backend=backend)["hr"]
        frame = out.get_frame(40)
        frames[backend] = np.stack([np.asarray(frame[p]) for p in range(3)], axis=-1).astype(np.float32)
    assert np.mean(np.abs(frames["cuda"] - frames["cpu"])) < 2.0
