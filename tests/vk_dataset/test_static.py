"""
Tests for the Denoise Dataset nodes: the sigma-clipped stack (mirroring
VSR-Dataset-Creator's tests), run detection on a synthetic video with gate
weave, and a full job writing hr/, lr/ and static_manifest.json.
"""

import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "include"))

from vk_dataset import run  # noqa: E402
from vk_dataset.nodes.static import zstack_average  # noqa: E402


def test_plain_mean_averages_every_frame():
    frames = [np.full((2, 2, 3), v, dtype=np.float32) for v in (0.0, 0.5, 1.0)]
    np.testing.assert_allclose(zstack_average(frames, 0.0), 0.5)


def test_sigma_clip_rejects_an_outlier():
    frames = [np.zeros((1, 1, 1), dtype=np.float32) for _ in range(8)]
    frames.append(np.ones((1, 1, 1), dtype=np.float32))
    np.testing.assert_allclose(zstack_average(frames, 0.0), 1.0 / 9.0)
    np.testing.assert_allclose(zstack_average(frames, 2.5), 0.0)


def make_video(folder):
    """24 static frames with weave and grain, 16 frames of motion, 24 static frames of another picture."""
    height, width = 144, 192
    rng = np.random.default_rng(3)

    def texture(seed):
        t = cv2.GaussianBlur(np.random.default_rng(seed).uniform(0, 1, (height + 8, width + 8, 3)).astype(np.float32),
                             (0, 0), 2.0)
        return (t - t.min()) / (t.max() - t.min())

    def shifted(t, dx, dy):
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(t, m, (t.shape[1], t.shape[0]), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)[4:4 + height, 4:4 + width]

    first, second = texture(1), texture(2)
    frames = [shifted(first, *rng.uniform(-1, 1, 2)) for _ in range(24)]
    frames += [shifted(second, i * 3.0, 0) for i in range(16)]
    frames += [shifted(second, *rng.uniform(-1, 1, 2)) for _ in range(24)]
    pictures = folder / "png"
    pictures.mkdir()
    for i, frame in enumerate(frames):
        noisy = np.clip(frame + rng.normal(0, 0.03, frame.shape), 0, 1)
        cv2.imwrite(str(pictures / f"{i:04d}.png"), (noisy * 255 + 0.5).astype(np.uint8)[:, :, ::-1])
    video = folder / "in.mkv"
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "24", "-i", str(pictures / "%04d.png"),
                        "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p", str(video)], check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("ffmpeg could not encode the test video")
    return video


def static_job(video, out, **find):
    nodes = [
        {"id": "src", "type": "load_video", "params": {"path": str(video)}},
        {"id": "find", "type": "find_static_runs", "params": {"window": 5, "min_gap": 3, "max_per_scene": 0, **find}},
        {"id": "centers", "type": "static_centers"},
        {"id": "stack", "type": "static_stack"},
        {"id": "out", "type": "write_static_dataset", "params": {"directory": str(out)}},
    ]
    edges = [("src.clip", "find.clip"), ("src.clip", "centers.clip"), ("find.runs", "centers.runs"),
             ("src.clip", "stack.clip"), ("find.runs", "stack.runs"), ("stack.clip", "out.hr"),
             ("centers.clip", "out.lr"), ("find.runs", "out.runs")]
    return {"version": 1, "seed": 1, "nodes": nodes, "edges": [{"from": a, "to": b} for a, b in edges]}


def test_static_runs_survive_weave_and_skip_motion(tmp_path):
    out = tmp_path / "out"
    result = run(static_job(make_video(tmp_path), out), emit=lambda _: None)
    assert result["pairs_exported"] == 2
    manifest = json.load(open(out / "static_manifest.json"))
    assert [(r["start_frame"], r["center_frame"]) for r in manifest["runs"]] == [(0, 2), (40, 42)]
    assert manifest["scenes_seen"] == 2
    assert sorted(os.listdir(out / "hr")) == sorted(os.listdir(out / "lr")) == ["static_000002.png",
                                                                               "static_000042.png"]
    assert manifest["settings"]["find"]["type"] == "find_static_runs"


def test_stack_has_less_grain_than_the_centre_frame(tmp_path):
    out = tmp_path / "out"
    run(static_job(make_video(tmp_path), out), emit=lambda _: None)

    def grain(path):
        image = cv2.imread(str(path)).astype(np.float32)
        return float(np.abs(image - cv2.GaussianBlur(image, (0, 0), 1.0)).mean())

    assert grain(out / "hr" / "static_000042.png") < grain(out / "lr" / "static_000042.png") * 0.8


def test_max_pairs_and_complexity_limit_the_runs(tmp_path):
    video = make_video(tmp_path)
    assert run(static_job(video, tmp_path / "a", max_pairs=1), emit=lambda _: None)["pairs_exported"] == 1
    with pytest.raises(ValueError, match="no static runs"):
        run(static_job(video, tmp_path / "b", min_complexity=10.0), emit=lambda _: None)


def test_presets_for_the_denoise_dataset_load():
    from vk_dataset import Graph
    Graph.load(os.path.join(os.path.dirname(__file__), "..", "..", "include", "vk_dataset", "presets",
                            "09-denoise-dataset-from-still-shots.json"))
