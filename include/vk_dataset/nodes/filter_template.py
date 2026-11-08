"""
Runs a filter template's VapourSynth code on a clip: one of Vapourkit's
.vkfilter templates (filter_catalog.py), or the user's edit of it.

The code runs as a template ran in Vapourkit: `clip`, `core` and `vs` are
bound, `original_clip` is the reference input (the clip itself when none is
connected), VK_BACKEND is the app's default AI backend, and whatever the code
leaves in `clip` is the result. Two names are new: `rng`, the node's seeded
generator, which notes every value it draws, and `note(key, value)` for
anything else worth recording in the manifest.
"""

import sys
from dataclasses import dataclass, field

import vapoursynth as vs

from ..filter_catalog import catalog
from ..media import convert_like
from ..registry import Param, node
from .model import resolve_backend

core = vs.core

DEFAULT_TEMPLATE = "CAS Sharpen"


@dataclass
class CodeParam(Param):
    """
    A code setting whose default follows a choice: while it is empty, the UI
    shows (and the node runs) the code of the template the choice names.
    """
    presets: dict = field(default_factory=dict)
    preset_from: str = ""

    def spec(self):
        return {**super().spec(), "presets": self.presets, "presetFrom": self.preset_from}


class NotedRandom:
    """The node's random.Random for template code; every value drawn is noted under "random", in order."""

    def __init__(self, ctx):
        self._ctx = ctx
        self._drawn = []

    def _keep(self, value):
        self._drawn.append(value if isinstance(value, (bool, int, float, str, type(None))) else repr(value))
        self._ctx.note("random", list(self._drawn))
        return value

    def random(self):
        return self._keep(self._ctx.rng.random())

    def uniform(self, low, high):
        return self._keep(self._ctx.rng.uniform(low, high))

    def randint(self, low, high):
        return self._keep(self._ctx.rng.randint(low, high))

    def choice(self, options):
        return self._keep(self._ctx.rng.choice(options))

    def gauss(self, mu=0.0, sigma=1.0):
        return self._keep(self._ctx.rng.gauss(mu, sigma))


def run_template(ctx, clip, code, name, reference=None):
    """Runs template code on clip and returns what it leaves in `clip`."""
    namespace = {
        "__name__": "__vk_filter_template__",
        "clip": clip, "core": core, "vs": vs,
        "original_clip": reference if reference is not None else clip,
        "VK_BACKEND": resolve_backend("auto")[0],
        "rng": NotedRandom(ctx), "note": ctx.note,
    }
    # Hybrid templates put a folder on sys.path before importing; it is put
    # back so that runs do not pile entries up.
    saved_path = list(sys.path)
    try:
        exec(compile(code, f"<filter template {name}>", "exec"), namespace)
    except Exception as error:
        raise RuntimeError(f"the {name} filter template failed: {error}") from error
    finally:
        sys.path[:] = saved_path
    result = namespace.get("clip")
    if not isinstance(result, vs.VideoNode):
        raise ValueError(f"the {name} filter template left no video in `clip`")
    return result


_TEMPLATES = catalog()


@node("filter_template", category="Transform", label="Filter Template", probability=True,
      description="Runs one of Vapourkit's VapourSynth filter templates, or your edit of it, on the clip.",
      inputs={"clip": "clip", "reference": "clip"}, optional_inputs=("reference",), outputs={"clip": "clip"},
      params=[
          Param("template", "choice", DEFAULT_TEMPLATE, choices=list(_TEMPLATES),
                help="The filter to run. Its code shows below, ready to edit."),
          CodeParam("code", "code", "", presets={name: t.code for name, t in _TEMPLATES.items()},
                    preset_from="template",
                    help="VapourSynth code that reads `clip` and assigns the result to it. Empty runs the "
                         "template as written. `rng` draws seeded random values, which the manifest records."),
          Param("keep_format", "bool", True,
                help="Convert the result back to the input's format (size and frame count are kept)."),
      ])
def filter_template(ctx, clip, reference=None, *, template, code, keep_format):
    if template not in _TEMPLATES:
        raise ValueError(f"there is no filter template named {template!r}")
    edited = bool(code.strip())
    ctx.note("template", template)
    if edited:
        ctx.note("edited", True)
    result = run_template(ctx, clip, code if edited else _TEMPLATES[template].code, template, reference)
    return {"clip": convert_like(result, clip) if keep_format else result}
