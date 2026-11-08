"""
vk_runtime.py - the helpers that pick a vs-mlrt backend for the app.

configure() records the backend the app chose and points vsmlrt at the files
the app provides; vk_backend() then builds a vsmlrt Backend for it, and the
dataset engine's model nodes call it when the app has configured it. The
dataset engine calls configure() once at start with the settings the app
passes it (vk_dataset/runtime_settings.py). The app copies this file into
data/config/, which is on the dataset engine's import path.
"""

import os
import shutil
import sys

import vapoursynth as vs

_backend = "tensorrt"
_backend_attrs = {}
_build_env = {}


def configure(backend, backend_attrs, build_env, models_path, trtexec_path, migraphx_driver_path=None):
    """
    Records the app's choices for vk_backend() and points vsmlrt at the files
    the app provides.

    backend         the app-selected backend id, which "auto" resolves to
    backend_attrs   backend id -> vsmlrt Backend attribute, from the app's registry
    build_env       environment seeded into a backend's custom_env; vsmlrt
                    builds TensorRT engines by spawning trtexec with a nearly
                    empty environment, and these are what the app's trtexec
                    shim needs to start
    models_path     the vsmlrt model zoo the app downloads (the pip wheels
                    ship no models folder)
    trtexec_path    the app's trtexec shim (pip TensorRT ships no trtexec)
    migraphx_driver_path
                    the migraphx-driver to run; when None it is
                    $VK_MIGRAPHX_DRIVER, then migraphx-driver on PATH, then
                    $ROCM_PATH/bin (the ROCm chosen in Settings names its own
                    driver, so one from another ROCm cannot pair with its
                    libraries)
    """
    global _backend, _backend_attrs, _build_env
    _backend = backend
    _backend_attrs = dict(backend_attrs)
    _build_env = dict(build_env)
    try:
        import vsmlrt
        vsmlrt.models_path = models_path
        vsmlrt.trtexec_path = trtexec_path
        # vsmlrt looks for migraphx-driver in a vsmlrt-hip folder beside the
        # first vs-mlrt plugin it finds, which a pip install doesn't have.
        vsmlrt.migraphx_driver_path = migraphx_driver_path or (
            os.environ.get("VK_MIGRAPHX_DRIVER")
            or shutil.which("migraphx-driver")
            or os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "bin", "migraphx-driver"))
    except Exception:
        pass


def configured():
    """Whether the app has called configure() in this process."""
    return bool(_backend_attrs)


def effective_backend(backend="auto"):
    """The backend id vk_backend() uses for this choice: the app's for "auto", else the one named."""
    return _backend if backend == "auto" else backend.lower()


def vk_backend(backend="auto", **kwargs):
    """
    A vsmlrt Backend for the app-selected backend, or for the one named;
    kwargs pass through (e.g. fp16=True).
    """
    from vsmlrt import Backend
    backend = effective_backend(backend)
    backend = {key: getattr(Backend, attr) for key, attr in _backend_attrs.items()}[backend](**kwargs)
    if hasattr(backend, "custom_env"):
        for key, value in _build_env.items():
            backend.custom_env.setdefault(key, value)
    # vsmlrt runs migraphx-driver with custom_env as its WHOLE environment,
    # so an empty one leaves HIP without PATH, HOME or its cache folders.
    if type(backend).__name__ == "MIGX":
        for key, value in os.environ.items():
            backend.custom_env.setdefault(key, value)
    return backend


def vk_migx_model(clips, network_path, input_name="input", fp16=True, num_streams=1):
    """
    Runs an ONNX model through vsmigx, which the MIGraphX provider's model step
    calls. vsmigx runs programs compiled for one input size, so one is compiled
    at the clip's resolution on first use - through vsmlrt's own
    migraphx_driver(), which caches it beside the ONNX keyed by size,
    precision, MIGraphX version, GPU and model checksum - and the compile is
    announced with the [vk-build] protocol so the app shows a banner.

    The clip is prepared in the float format of the model's import precision,
    but vsmigx demands the program's I/O type, so on a mismatch the call is
    retried once in the other float format.
    """
    import vsmlrt
    core = vs.core
    clips = clips if isinstance(clips, list) else [clips]
    width, height = clips[0].width, clips[0].height
    program = dict(network_path=network_path, opt_shapes=(width, height), fp16=fp16,
                   fast_math=True, exhaustive_tune=False, device_id=0, short_path=None)
    compiled = os.access(vsmlrt.get_mxr_path(**program), os.R_OK)
    label = f"Compiling MIGraphX program: {os.path.basename(network_path)} at {width}x{height}"
    if not compiled:
        print(f"[vk-build] begin {label}", file=sys.stderr, flush=True)
    try:
        mxr = vsmlrt.migraphx_driver(channels=sum(c.format.num_planes for c in clips), input_name=input_name,
                                     custom_env=dict(os.environ), **program)
    finally:
        if not compiled:
            print(f"[vk-build] end {label}", file=sys.stderr, flush=True)
    try:
        return core.migx.Model(clips, mxr, num_streams=num_streams)
    except vs.Error as error:
        # "sample type mismatch" / "bytes per sample mismatch"
        if "sample" not in str(error) or "mismatch" not in str(error):
            raise
        bits = 32 if clips[0].format.bits_per_sample == 16 else 16
        clips = [core.resize.Point(c, format=c.format.replace(bits_per_sample=bits)) for c in clips]
        return core.migx.Model(clips, mxr, num_streams=num_streams)
