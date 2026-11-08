"""
The inference settings the app hands the engine: its default backend, the
registry's backend id -> vsmlrt Backend attribute map, the environment
TensorRT engine builds need, and the paths of the trtexec shim, the model
folder and migraphx-driver. The app passes them as JSON in the
VK_DATASET_RUNTIME environment variable, and the engine applies them once at
start, before any model node builds a backend.
"""

import json
import os

RUNTIME_ENV = "VK_DATASET_RUNTIME"


def apply_runtime_settings(environ=os.environ):
    """
    Configures vk_runtime from VK_DATASET_RUNTIME and returns the settings, or
    None when the variable is not set (the engine run on its own).
    """
    raw = environ.get(RUNTIME_ENV)
    if not raw:
        return None
    settings = json.loads(raw)
    import vk_runtime
    vk_runtime.configure(settings["backend"], settings["backend_attrs"], settings.get("build_env", {}),
                         settings.get("models_path"), settings.get("trtexec_path"),
                         settings.get("migraphx_driver_path"))
    return settings
