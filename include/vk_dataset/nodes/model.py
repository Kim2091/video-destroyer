"""
AI model nodes. Image models run as ONNX through vs-mlrt, so they can use
any of the app's inference backends and need no PyTorch. scripts/export_dataset_models.py converts the training
checkpoints to ONNX; the models themselves are referenced by path and are not
part of this repository.

AniCC reads the HR and LR picture at their own sizes in one pass, which a
vs-mlrt model (one input size for all its inputs) cannot, so it runs on
onnxruntime directly.
"""

import os
import sys

import numpy as np
import vapoursynth as vs

from ..media import frame_array
from ..registry import Param, node
from .hr_process import _copy_with, rgb_float

core = vs.core

BACKENDS = ["auto", "tensorrt", "directml", "ncnn", "migraphx", "cpu"]

BACKEND_LABELS = {"tensorrt": "TensorRT", "directml": "DirectML", "ncnn": "NCNN Vulkan", "migraphx": "MIGraphX",
                  "cpu": "the CPU (ONNX Runtime)"}


def resolve_backend(name):
    """
    The backend a model node runs on, and why. "auto" follows the app's
    Default AI backend setting, which the app hands vk_runtime at start; a
    backend named on the node wins over it. Run without the app, "auto" is
    DirectML on Windows and the CPU elsewhere, the providers the pip wheels
    carry.
    """
    if name != "auto":
        return name, "chosen on the node"
    try:
        import vk_runtime
        if vk_runtime.configured():
            return vk_runtime.effective_backend("auto"), "the app's default AI backend"
    except ImportError:
        pass
    return ("directml" if sys.platform == "win32" else "cpu"), "no app backend setting"


def make_backend(name, fp16, num_streams):
    """A vsmlrt Backend for a backend id from resolve_backend()."""
    import vsmlrt
    if name != "cpu":
        try:
            import vk_runtime
            if vk_runtime.configured():
                return vk_runtime.vk_backend(name, fp16=fp16, num_streams=num_streams)
        except ImportError:
            pass
    attribute = {"tensorrt": "TRT", "directml": "ORT_DML", "ncnn": "NCNN_VK", "migraphx": "MIGX",
                 "cpu": "ORT_CPU"}[name]
    return getattr(vsmlrt.Backend, attribute)(fp16=fp16, num_streams=num_streams)


def announce_backend(ctx, model, backend, reason):
    """Says once per node which backend it runs its model on, in the run log and the preview's messages."""
    state = ctx.state()
    if state.get("announced") == (model, backend):
        return
    state["announced"] = (model, backend)
    text = (f"{ctx.node.label} ({ctx.node.id}) runs {os.path.basename(model)} on "
            f"{BACKEND_LABELS.get(backend, backend)} ({reason})")
    if backend == "tensorrt":
        text += "; the first run at a new frame size builds a TensorRT engine, which can take a few minutes"
    ctx.message(text)


def _precision(clip, fp16):
    if clip.format.color_family == vs.GRAY:
        return vs.GRAYH if fp16 else vs.GRAYS
    return vs.RGBH if fp16 else vs.RGBS


def infer(ctx, clips, model, *, backend, fp16, num_streams, tile_width=0, tile_height=0, overlap=0):
    """
    Runs an ONNX model over RGB clips (their planes become the model's input
    channels, in order) and returns float RGB. A tile size of 0 runs whole
    frames; tiles overlap by `overlap` pixels and only their centres are kept.
    """
    import vsmlrt
    backend, reason = resolve_backend(backend)
    announce_backend(ctx, model, backend, reason)
    clips = [core.resize.Point(c, format=_precision(c, fp16)) for c in clips]
    first = clips[0]
    tile = (tile_width or first.width, tile_height or first.height)
    out = vsmlrt.inference(clips, model, overlap=(overlap, overlap), tilesize=tile,
                           backend=make_backend(backend, fp16, num_streams))
    return rgb_float(out)


def model_params(model_help, tile, overlap):
    return [
        Param("model", "path", "", help=model_help),
        Param("backend", "choice", "auto", choices=BACKENDS, help="'auto' follows the app's backend setting."),
        Param("fp16", "bool", True, help="Run the model in half precision."),
        Param("tile_width", "int", tile, min=0, help="Tile width in pixels (0 = the whole frame)."),
        Param("tile_height", "int", tile, min=0, help="Tile height in pixels (0 = the whole frame)."),
        Param("overlap", "int", overlap, min=0, help="Pixels tiles overlap by; a tile contributes only its centre."),
        Param("num_streams", "int", 1, min=1, help="Frames inferred at once."),
    ]


@node("onnx_model", category="AI Model", label="ONNX Model",
      description="Runs any single-input ONNX image model through vs-mlrt.",
      inputs={"clip": "clip"}, outputs={"clip": "clip"}, probability=True,
      params=model_params("An ONNX model file.", 0, 0))
def onnx_model(ctx, clip, *, model, backend, fp16, tile_width, tile_height, overlap, num_streams):
    require_model(model)
    return {"clip": infer(ctx, [rgb_float(clip)], model, backend=backend, fp16=fp16, num_streams=num_streams,
                          tile_width=tile_width, tile_height=tile_height, overlap=overlap)}


def require_model(path):
    if not path:
        raise ValueError("no model file is set")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"the model file {path} does not exist")


@node("denoise_hr", category="HR Process", label="Denoise HR",
      description="Cleans the HR image with a 1x ONNX denoising model, such as the exported AnimeDenoise or GaterV3.",
      inputs={"clip": "clip"}, outputs={"clip": "clip"}, probability=True,
      params=model_params("A 1x denoising model as ONNX.", 512, 32))
def denoise_hr(ctx, clip, *, model, backend, fp16, tile_width, tile_height, overlap, num_streams):
    require_model(model)
    source = rgb_float(clip)
    out = infer(ctx, [source], model, backend=backend, fp16=fp16, num_streams=num_streams,
                tile_width=tile_width, tile_height=tile_height, overlap=overlap)
    if (out.width, out.height) != (source.width, source.height):
        raise ValueError(f"the denoise model must be 1x, but it turns {source.width}x{source.height} "
                         f"into {out.width}x{out.height}")
    return {"clip": out}


def reflect_pad(clip, right, bottom):
    """Pads the right and bottom edges by mirroring the picture without repeating its edge row or column."""
    if right:
        edge = core.std.FlipHorizontal(core.std.Crop(clip, left=clip.width - 1 - right, right=1))
        clip = core.std.StackHorizontal([clip, edge])
    if bottom:
        edge = core.std.FlipVertical(core.std.Crop(clip, top=clip.height - 1 - bottom, bottom=1))
        clip = core.std.StackVertical([clip, edge])
    return clip


@node("prism_deh264", category="HR Process", label="Prism de-H264",
      description="Removes H.264 compression damage with a Prism SCUNet ONNX model, which takes the RGB "
                  "picture plus a constant plane holding the CRF strength / 51.",
      inputs={"clip": "clip"}, outputs={"clip": "clip"}, probability=True,
      params=[Param("model", "path", "", help="A Prism de-H264 model as ONNX."),
              Param("strength_crf", "float", 20.0, min=0.0, max=51.0, help="The CRF the damage is assumed to have."),
              Param("backend", "choice", "auto", choices=BACKENDS, help="'auto' follows the app's backend setting."),
              Param("fp16", "bool", True, help="Run the model in half precision."),
              Param("tile_size", "int", 512, min=0, help="Tile size, a multiple of 64 (0 = the whole frame)."),
              Param("overlap", "int", 64, min=0, help="Pixels tiles overlap by."),
              Param("num_streams", "int", 1, min=1, help="Frames inferred at once.")])
def prism_deh264(ctx, clip, *, model, strength_crf, backend, fp16, tile_size, overlap, num_streams):
    require_model(model)
    if tile_size % 64:
        raise ValueError("the Prism tile size must be a multiple of 64")
    rgb = rgb_float(clip)
    width, height = rgb.width, rgb.height
    # The model wants sizes in multiples of 64.
    padded = reflect_pad(rgb, -width % 64, -height % 64)
    strength = core.std.BlankClip(padded, format=vs.GRAYS, color=[float(strength_crf) / 51.0])
    out = infer(ctx, [padded, strength], model, backend=backend, fp16=fp16, num_streams=num_streams,
                tile_width=tile_size, tile_height=tile_size, overlap=overlap)
    return {"clip": core.std.Crop(out, right=out.width - width, bottom=out.height - height)}


def _provider_list():
    import onnxruntime
    available = onnxruntime.get_available_providers()
    return [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider") if p in available]


@node("anicc", category="HR Process", label="AniCC Colour Correction",
      description="Matches the HR image's colours to the LR image with the AniCC (BGCC v4) ONNX model. "
                  "The model reads both pictures whole, at their own sizes.",
      inputs={"hr": "clip", "lr": "clip"}, outputs={"hr": "clip", "lr": "clip"}, probability=True,
      params=[Param("model", "path", "", help="The AniCC model as ONNX (exported with scripts/export_dataset_models.py).")])
def anicc(ctx, hr, lr, *, model):
    require_model(model)
    try:
        import onnxruntime
    except ImportError as e:
        raise RuntimeError("AniCC runs on onnxruntime, which is not installed in this Python") from e
    hr, lr = rgb_float(hr), rgb_float(lr)
    session = ctx.state().get("session")
    if session is None:
        session = ctx.state()["session"] = onnxruntime.InferenceSession(model, providers=_provider_list())
    hr_name, lr_name = (i.name for i in session.get_inputs()[:2])
    output = session.get_outputs()[0].name

    def correct(n, f):
        image = frame_array(f[0])
        height, width = image.shape[:2]
        # The encoder halves the picture twice, so the model wants multiples of 4.
        padded = np.pad(image, ((0, -height % 4), (0, -width % 4), (0, 0)), mode="reflect")
        result = session.run([output], {
            hr_name: np.ascontiguousarray(padded.transpose(2, 0, 1)[None]),
            lr_name: np.ascontiguousarray(frame_array(f[1]).transpose(2, 0, 1)[None]),
        })[0][0]
        result = np.clip(result.transpose(1, 2, 0)[:height, :width], 0.0, 1.0)
        return _copy_with(f[0], [result[:, :, p] for p in range(3)])

    return {"hr": core.std.ModifyFrame(hr, [hr, lr], correct), "lr": lr}
