"""
Tests for the HR Process nodes: the Match Gamma estimator, loading a dataset
folder, the colour and filter nodes, the processed-dataset writer, and the
model nodes (skipped when no model or backend is available). Mirrors
VSR-Dataset-Creator's Match Gamma tests where they apply.
"""

import json
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "include"))

import vk_dataset  # noqa: E402
from vk_dataset import match_gamma, run  # noqa: E402
from vk_dataset.nodes.dataset_source import read_image  # noqa: E402
from vk_dataset.nodes.output import write_image  # noqa: E402


def noise(seed, shape=(48, 48, 3), low=0.1, high=0.9):
    return np.random.default_rng(seed).uniform(low, high, size=shape).astype(np.float32)


def u8(image):
    return (np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8)


def put(root, side, name, image):
    os.makedirs(root / side, exist_ok=True)
    write_image(str(root / side / name), u8(image), "png", 0)


def mean_abs(root, side, name, reference):
    return float(np.abs(read_image(str(root / side / name)).astype(np.float32) / 255 - reference).mean())


def job(nodes, edges):
    return {"version": 1, "seed": 1, "nodes": nodes, "edges": [{"from": a, "to": b} for a, b in edges]}


def process(root, steps=(), gamma=None, lr_all=False):
    """load_dataset -> optional match_gamma -> steps (each [id, type, params]) -> write_processed."""
    nodes = [{"id": "src", "type": "load_dataset", "params": {"directory": str(root)}},
             {"id": "out", "type": "write_processed", "params": {"directory": str(root)}}]
    edges, last = [], "src.hr"
    if gamma:
        nodes.append({"id": "mg", "type": "match_gamma", "params": gamma})
        edges += [("src.hr", "mg.hr"), ("src.lr", "mg.lr")]
        last = "mg.hr"
        if lr_all:
            edges += [("src.lr_all", "mg.lr_all"), ("mg.lr_all", "out.lr_all")]
    for node_id, node_type, params in steps:
        nodes.append({"id": node_id, "type": node_type, "params": params})
        edges.append((last, f"{node_id}.clip"))
        last = f"{node_id}.clip"
    edges.append((last, "out.hr"))
    return run(job(nodes, edges), emit=lambda _: None)


# --- the estimator ---------------------------------------------------------

def test_estimates_known_gamma():
    reference = noise(7)
    assert match_gamma.estimate(reference ** 2.0, reference) == pytest.approx(0.5, abs=0.01)


def test_identical_images_estimate_unity():
    image = noise(11)
    assert match_gamma.estimate(image, image) == pytest.approx(1.0, abs=1e-4)


def test_near_unity_gamma_is_identity():
    image = noise(3)
    assert match_gamma.apply(image, 1.0) is image


def test_apply_then_estimate_round_trips():
    reference = noise(5)
    image = reference ** 3.0
    matched = match_gamma.apply(image, match_gamma.estimate(image, reference))
    assert match_gamma.estimate(matched, reference) == pytest.approx(1.0, abs=0.01)


# --- loading a dataset -----------------------------------------------------

def test_node_specs_list_the_hr_process_nodes():
    types = {s["type"] for s in vk_dataset.node_specs()}
    assert {"load_dataset", "denoise_hr", "fast_line_darken", "fine_dehalo", "prism_deh264", "match_gamma",
            "tone_adjust", "anicc", "onnx_model", "write_processed"} <= types
    json.dumps(vk_dataset.node_specs())


def test_presets_load():
    folder = os.path.join(os.path.dirname(__file__), "..", "..", "include", "vk_dataset", "presets")
    names = [n for n in os.listdir(folder) if n.endswith(".json")]
    assert len(names) >= 3
    for name in names:
        vk_dataset.Graph.load(os.path.join(folder, name))


def test_one_target_per_sequence_is_selected(tmp_path):
    base = noise(1)
    for frame in range(1, 6):
        put(tmp_path, "lr", f"show1_Frame{frame}.png", base)
    for frame in range(1, 6):  # an older dataset: HR for every frame
        put(tmp_path, "hr", f"show1_Frame{frame}.png", base)
    put(tmp_path, "hr", "flat.png", base)
    put(tmp_path, "lr", "flat.png", base)
    result = process(tmp_path)
    assert result["hr_processed"] == 2
    assert sorted(os.listdir(tmp_path / "hr_p")) == ["flat.png", "show1_Frame3.png"]


def test_even_sequence_length_is_rejected(tmp_path):
    for frame in range(1, 5):
        put(tmp_path, "lr", f"show1_Frame{frame}.png", noise(frame))
    put(tmp_path, "hr", "show1_Frame2.png", noise(1))
    with pytest.raises(ValueError, match="odd length"):
        process(tmp_path)


def test_originals_are_left_alone(tmp_path):
    put(tmp_path, "hr", "a.png", noise(2))
    before = (tmp_path / "hr" / "a.png").read_bytes()
    process(tmp_path, [("t", "tone_adjust", {"brightness": 0.1})])
    assert (tmp_path / "hr" / "a.png").read_bytes() == before
    assert not (tmp_path / "hr_p" / ".incomplete.json").exists()
    manifest = json.load(open(tmp_path / "processed_manifest.json"))
    assert manifest["files"]["hr"] == ["a.png"]


def test_missing_lr_is_reported_only_when_a_node_needs_it(tmp_path):
    put(tmp_path, "hr", "a.png", noise(2))
    process(tmp_path)
    with pytest.raises(ValueError, match="no lr folder"):
        process(tmp_path, gamma={"direction": "hr_to_lr"})


def test_output_folder_cannot_be_an_input_folder(tmp_path):
    put(tmp_path, "hr", "a.png", noise(2))
    raw = job([{"id": "src", "type": "load_dataset", "params": {"directory": str(tmp_path)}},
               {"id": "out", "type": "write_processed", "params": {"directory": str(tmp_path), "hr_folder": "HR"}}],
              [("src.hr", "out.hr")])
    with pytest.raises(ValueError, match="new folder name"):
        run(raw, emit=lambda _: None)


# --- colour nodes ----------------------------------------------------------

def test_hr_is_matched_to_lr_per_scene(tmp_path):
    base = noise(21)
    put(tmp_path, "lr", "frame001.png", base)
    put(tmp_path, "hr", "frame001.png", base ** 0.5)
    process(tmp_path, gamma={"direction": "hr_to_lr"})
    assert mean_abs(tmp_path, "hr_p", "frame001.png", base) < 0.02
    manifest = json.load(open(tmp_path / "processed_manifest.json"))
    assert manifest["settings"]["mg"]["params"]["scene_gammas"]["frame001"] == pytest.approx(2.0, abs=0.05)
    assert not (tmp_path / "lr_p").exists()


def test_lr_to_hr_writes_the_whole_adjusted_lr_sequence(tmp_path):
    base = noise(33)
    put(tmp_path, "hr", "show1_Frame3.png", base)
    for frame in range(1, 6):
        put(tmp_path, "lr", f"show1_Frame{frame}.png", base ** 2.0)
    result = process(tmp_path, gamma={"direction": "lr_to_hr"}, lr_all=True)
    assert result["lr_adjusted"] == 5
    assert sorted(os.listdir(tmp_path / "lr_p")) == [f"show1_Frame{n}.png" for n in range(1, 6)]
    for n in range(1, 6):
        assert mean_abs(tmp_path, "lr_p", f"show1_Frame{n}.png", base) < 0.02
    assert mean_abs(tmp_path, "hr_p", "show1_Frame3.png", base) < 0.01


def test_show_scope_pools_one_gamma(tmp_path):
    for index, seed in ((1, 41), (2, 43)):
        base = noise(seed)
        put(tmp_path, "hr", f"show{index}_Frame2.png", base)
        for frame in range(1, 4):
            put(tmp_path, "lr", f"show{index}_Frame{frame}.png", base ** 2.0)
    result = process(tmp_path, gamma={"direction": "lr_to_hr", "scope": "show"}, lr_all=True)
    assert result["lr_adjusted"] == 6
    manifest = json.load(open(tmp_path / "processed_manifest.json"))
    assert manifest["settings"]["mg"]["params"]["show_gamma"] == pytest.approx(0.5, abs=0.02)


def test_gamma_is_clamped(tmp_path):
    base = noise(55)
    put(tmp_path, "lr", "a.png", base)
    put(tmp_path, "hr", "a.png", base ** 0.5)
    process(tmp_path, gamma={"direction": "hr_to_lr", "max_gamma": 1.5})
    manifest = json.load(open(tmp_path / "processed_manifest.json"))
    assert manifest["settings"]["mg"]["params"]["scene_gammas"]["a"] == 1.5


def test_tone_adjust_matches_the_formula(tmp_path):
    image = noise(9)
    put(tmp_path, "hr", "a.png", image)
    process(tmp_path, [("t", "tone_adjust", {"brightness": 0.1, "contrast": 0.3, "gamma": 1.5})])
    source = u8(image).astype(np.float32) / 255
    slope = np.tan((0.3 + 1) * np.pi / 4)
    expected = np.clip(np.clip((source - 0.5) * slope + 0.5 + 0.1, 0, 1) ** (1 / 1.5), 0, 1)
    assert mean_abs(tmp_path, "hr_p", "a.png", expected) < 0.005


# --- filters ---------------------------------------------------------------

def lined_picture():
    image = np.full((96, 96, 3), 0.8, np.float32)
    image[:, 46:48] = 0.55
    image[40:42, :] = 0.55
    return image


def test_fast_line_darken_darkens_lines_and_keeps_flat_areas(tmp_path):
    pytest.importorskip("vsrgtools")
    image = lined_picture()
    put(tmp_path, "hr", "a.png", image)
    process(tmp_path, [("f", "fast_line_darken", {})])
    out = read_image(str(tmp_path / "hr_p" / "a.png")).astype(np.float32) / 255
    assert out[:, 46:48].mean() < image[:, 46:48].mean() - 0.02
    assert np.abs(out[10:30, 10:30] - 0.8).max() < 0.01


def test_fine_dehalo_runs_and_keeps_colour(tmp_path):
    pytest.importorskip("vsdehalo")
    image = lined_picture()
    image[..., 0] *= 0.6
    put(tmp_path, "hr", "a.png", image)
    process(tmp_path, [("d", "fine_dehalo", {"exclude_strength": 0.5})])
    out = read_image(str(tmp_path / "hr_p" / "a.png")).astype(np.float32) / 255
    assert out.shape == image.shape
    assert abs(float((out[..., 0] - out[..., 1]).mean()) - float((image[..., 0] - image[..., 1]).mean())) < 0.02


def test_disabled_node_passes_through(tmp_path):
    image = noise(4)
    put(tmp_path, "hr", "a.png", image)
    raw = job([{"id": "src", "type": "load_dataset", "params": {"directory": str(tmp_path)}},
               {"id": "t", "type": "tone_adjust", "params": {"brightness": 0.5}, "enabled": False},
               {"id": "out", "type": "write_processed", "params": {"directory": str(tmp_path)}}],
              [("src.hr", "t.clip"), ("t.clip", "out.hr")])
    run(raw, emit=lambda _: None)
    assert mean_abs(tmp_path, "hr_p", "a.png", u8(image).astype(np.float32) / 255) == 0


# --- models ----------------------------------------------------------------

MODELS = os.environ.get("VK_DATASET_TEST_MODELS", "")
CPU_MODEL = {"fp16": False, "backend": "cpu", "tile_width": 0, "tile_height": 0, "overlap": 0}


def onnx_path(name):
    path = os.path.join(MODELS, name)
    if not MODELS or not os.path.isfile(path):
        pytest.skip(f"set VK_DATASET_TEST_MODELS to a folder with {name}")
    return path


def vsmlrt_or_skip():
    pytest.importorskip("onnx")
    # vsmlrt.py (include/) refuses to import without a vs-mlrt plugin.
    import vapoursynth as vs
    if not hasattr(vs.core, "ort"):
        pytest.skip("the vsort plugin is not installed")
    pytest.importorskip("vsmlrt")


def test_denoise_node_runs_a_1x_onnx_model_through_vsmlrt(tmp_path):
    vsmlrt_or_skip()
    import onnx
    from onnx import TensorProto, helper
    # A 1x model that halves its input, built here so no model file is needed.
    graph = helper.make_graph(
        [helper.make_node("Mul", ["input", "half"], ["output"])], "half",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "h", "w"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, "h", "w"])],
        [helper.make_tensor("half", TensorProto.FLOAT, [], [0.5])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    path = tmp_path / "half.onnx"
    onnx.save(model, str(path))
    image = noise(8)
    put(tmp_path, "hr", "a.png", image)
    process(tmp_path, [("m", "denoise_hr", {"model": str(path), **CPU_MODEL})])
    assert mean_abs(tmp_path, "hr_p", "a.png", u8(image).astype(np.float32) / 255 / 2) < 0.01


def test_denoise_node_rejects_a_model_that_resizes(tmp_path):
    vsmlrt_or_skip()
    import onnx
    from onnx import TensorProto, helper
    graph = helper.make_graph(
        [helper.make_node("Resize", ["input", "", "scales"], ["output"], mode="nearest")], "up",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "h", "w"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, "h2", "w2"])],
        [helper.make_tensor("scales", TensorProto.FLOAT, [4], [1, 1, 2, 2])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    path = tmp_path / "up.onnx"
    onnx.save(model, str(path))
    put(tmp_path, "hr", "a.png", noise(8))
    with pytest.raises(ValueError, match="must be 1x"):
        process(tmp_path, [("m", "denoise_hr", {"model": str(path), **CPU_MODEL})])


def test_model_node_needs_a_file(tmp_path):
    put(tmp_path, "hr", "a.png", noise(8))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        process(tmp_path, [("m", "onnx_model", {"model": str(tmp_path / "none.onnx")})])


def test_exported_denoiser_runs_through_the_node(tmp_path):
    vsmlrt_or_skip()
    model = onnx_path("1x_bndl_animedenoise_v3.onnx")
    put(tmp_path, "hr", "a.png", noise(8, (64, 64, 3)))
    process(tmp_path, [("m", "denoise_hr", {"model": model, **CPU_MODEL})])
    assert (tmp_path / "hr_p" / "a.png").exists()


def test_exported_prism_runs_through_the_node(tmp_path):
    vsmlrt_or_skip()
    model = onnx_path("prism_h264_scunet.onnx")
    put(tmp_path, "hr", "a.png", noise(8, (70, 90, 3)))
    process(tmp_path, [("m", "prism_deh264", {"model": model, "backend": "cpu", "fp16": False, "tile_size": 0})])
    assert read_image(str(tmp_path / "hr_p" / "a.png")).shape == (70, 90, 3)


def test_exported_anicc_runs_on_two_sizes(tmp_path):
    pytest.importorskip("onnxruntime")
    model = onnx_path("2x_bndl_anicc.onnx")
    base = noise(8, (64, 64, 3))
    put(tmp_path, "hr", "a.png", base)
    put(tmp_path, "lr", "a.png", cv2.resize(base, (32, 32), interpolation=cv2.INTER_AREA))
    raw = job([{"id": "src", "type": "load_dataset", "params": {"directory": str(tmp_path)}},
               {"id": "c", "type": "anicc", "params": {"model": model}},
               {"id": "out", "type": "write_processed", "params": {"directory": str(tmp_path)}}],
              [("src.hr", "c.hr"), ("src.lr", "c.lr"), ("c.hr", "out.hr")])
    run(raw, emit=lambda _: None)
    assert read_image(str(tmp_path / "hr_p" / "a.png")).shape == (64, 64, 3)
