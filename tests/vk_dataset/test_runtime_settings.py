"""
Tests for the inference settings the app hands the engine: VK_DATASET_RUNTIME
reaches vk_runtime.configure() unchanged, a model node set to "auto" follows
the app's backend, and a backend named on the node wins over it.
"""

import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "include"))

import vk_runtime  # noqa: E402
from vk_dataset.nodes import model  # noqa: E402
from vk_dataset.runtime_settings import RUNTIME_ENV, apply_runtime_settings  # noqa: E402

SETTINGS = {
    "backend": "tensorrt",
    "backend_attrs": {"tensorrt": "TRT", "directml": "ORT_DML", "ncnn": "NCNN_VK", "migraphx": "MIGX"},
    "build_env": {"SystemRoot": "C:/Windows"},
    "models_path": "C:/vd/data/models",
    "trtexec_path": "C:/vd/data/trtexec.cmd",
    "migraphx_driver_path": None,
}


class FakeBackend:
    """Stands in for each vsmlrt.Backend class: records which one was built and with what."""

    def __init__(self, name, **kwargs):
        self.name, self.kwargs, self.custom_env = name, kwargs, {}


@pytest.fixture
def fake_vsmlrt(monkeypatch):
    backend = types.SimpleNamespace(**{attr: (lambda attr=attr, **kw: FakeBackend(attr, **kw))
                                       for attr in ("TRT", "ORT_DML", "NCNN_VK", "MIGX", "ORT_CPU")})
    module = types.SimpleNamespace(Backend=backend)
    monkeypatch.setitem(sys.modules, "vsmlrt", module)
    yield module


@pytest.fixture
def unconfigured(monkeypatch):
    monkeypatch.setattr(vk_runtime, "_backend", "tensorrt")
    monkeypatch.setattr(vk_runtime, "_backend_attrs", {})
    monkeypatch.setattr(vk_runtime, "_build_env", {})


def test_settings_reach_configure_unchanged(monkeypatch, unconfigured):
    calls = []
    monkeypatch.setattr(vk_runtime, "configure", lambda *args: calls.append(args))
    assert apply_runtime_settings({RUNTIME_ENV: json.dumps(SETTINGS)}) == SETTINGS
    assert calls == [("tensorrt", SETTINGS["backend_attrs"], SETTINGS["build_env"], SETTINGS["models_path"],
                      SETTINGS["trtexec_path"], None)]


def test_no_settings_leave_vk_runtime_alone(monkeypatch, unconfigured):
    monkeypatch.setattr(vk_runtime, "configure", lambda *args: pytest.fail("configure was called"))
    assert apply_runtime_settings({}) is None
    assert not vk_runtime.configured()


def test_configure_points_vsmlrt_at_the_app_files(fake_vsmlrt, unconfigured):
    apply_runtime_settings({RUNTIME_ENV: json.dumps(SETTINGS)})
    assert fake_vsmlrt.trtexec_path == SETTINGS["trtexec_path"]
    assert fake_vsmlrt.models_path == SETTINGS["models_path"]


def test_auto_follows_the_app_backend(fake_vsmlrt, unconfigured):
    apply_runtime_settings({RUNTIME_ENV: json.dumps(SETTINGS)})
    name, reason = model.resolve_backend("auto")
    assert (name, reason) == ("tensorrt", "the app's default AI backend")
    backend = model.make_backend(name, True, 1)
    assert backend.name == "TRT" and backend.kwargs == {"fp16": True, "num_streams": 1}
    # The trtexec shim's environment rides along into TensorRT builds.
    assert backend.custom_env == SETTINGS["build_env"]


def test_a_backend_named_on_the_node_wins(fake_vsmlrt, unconfigured):
    apply_runtime_settings({RUNTIME_ENV: json.dumps({**SETTINGS, "backend": "tensorrt"})})
    name, reason = model.resolve_backend("directml")
    assert (name, reason) == ("directml", "chosen on the node")
    assert model.make_backend(name, False, 2).name == "ORT_DML"
    assert model.make_backend("cpu", False, 1).name == "ORT_CPU"


def test_without_the_app_auto_is_what_the_wheels_carry(fake_vsmlrt, unconfigured):
    name, reason = model.resolve_backend("auto")
    assert name == ("directml" if sys.platform == "win32" else "cpu")
    assert reason == "no app backend setting"
    assert model.make_backend(name, True, 1).name == ("ORT_DML" if sys.platform == "win32" else "ORT_CPU")


def test_the_run_log_says_once_which_backend_a_node_uses():
    messages = []
    node = types.SimpleNamespace(id="m", label="Denoise HR")
    state = {}
    ctx = types.SimpleNamespace(node=node, state=lambda: state, message=messages.append)
    for _ in range(3):
        model.announce_backend(ctx, "C:/models/clean.onnx", "tensorrt", "the app's default AI backend")
    assert len(messages) == 1
    assert messages[0].startswith("Denoise HR (m) runs clean.onnx on TensorRT (the app's default AI backend)")
    assert "builds a TensorRT engine" in messages[0]
    model.announce_backend(ctx, "C:/models/clean.onnx", "directml", "chosen on the node")
    assert messages[-1] == "Denoise HR (m) runs clean.onnx on DirectML (chosen on the node)"
