"""
Tests for the filter template catalog and the Filter Template node. The
templates that call plugins or scripts setup installs are skipped where
those are missing. Run from the repo root:

    python -m pytest tests/vk_dataset
"""

import importlib.util
import os
import random
import sys

import pytest
import vapoursynth as vs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "include"))

from vk_dataset import NODE_TYPES  # noqa: E402
from vk_dataset.filter_catalog import catalog, render  # noqa: E402
from vk_dataset.media import load  # noqa: E402

core = vs.core
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "two_scenes.mkv")


class FakeContext:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.notes = {}

    def note(self, key, value):
        self.notes[key] = value


def run(clip, seed=0, reference=None, **params):
    values = {p.name: p.default for p in NODE_TYPES["filter_template"].params}
    values.update(params)
    ctx = FakeContext(seed)
    return NODE_TYPES["filter_template"].func(ctx, clip, reference, **values)["clip"], ctx


@pytest.fixture(scope="module")
def clip():
    return load(FIXTURE)[:8]


def test_every_template_loads_as_plain_python():
    templates = catalog()
    assert len(templates) > 100
    for name, template in templates.items():
        assert "{{" not in template.code, name
        compile(template.code, name, "exec")
    assert "Apply LUT" in templates
    # Templates that need the upscaling UI, a step reference or something setup no longer installs.
    for dropped in ("Color Grade", "Wavelet Color Fix from Step", "DLSS Neural Uplift", "Grain Synth", "RIFE",
                    "Retinex Edge Mask"):
        assert dropped not in templates


def test_render_writes_declared_defaults_as_python_literals():
    variables = {"path": {"default": "C:\\a \"b\""}, "n": {"default": 2}, "on": {"default": True}}
    assert render("x = {{path}}\ny = {{ n }}\nz = {{on}}\nw = {{other}}", variables) == \
        'x = "C:\\\\a \\"b\\""\ny = 2\nz = True\nw = {{other}}'


def test_the_spec_offers_every_template_with_its_code():
    params = {p["name"]: p for p in NODE_TYPES["filter_template"].spec()["params"]}
    assert params["template"]["choices"] == list(catalog())
    assert params["template"]["default"] in catalog()
    assert params["code"]["presetFrom"] == "template"
    assert params["code"]["presets"] == {name: t.code for name, t in catalog().items()}


def test_edited_code_runs_with_a_seeded_rng_whose_draws_are_noted(clip):
    code = "strength = rng.uniform(0.2, 0.8)\nnote('strength', strength)\nclip = core.std.Invert(clip)"
    out, ctx = run(clip, seed=3, code=code)
    drawn = random.Random(3).uniform(0.2, 0.8)
    assert ctx.notes["random"] == [drawn]
    assert ctx.notes["strength"] == drawn
    assert ctx.notes["edited"] is True
    assert out.format.id == clip.format.id and out.num_frames == clip.num_frames


def test_the_result_is_converted_back_to_the_input_format(clip):
    code = "clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s='709')"
    assert run(clip, code=code)[0].format.id == clip.format.id
    assert run(clip, code=code, keep_format=False)[0].format.id == vs.RGBS


def test_code_that_leaves_no_clip_is_refused(clip):
    with pytest.raises(ValueError, match="no video"):
        run(clip, code="clip = None")
    with pytest.raises(RuntimeError, match="CAS Sharpen filter template failed"):
        run(clip, code="raise ValueError('boom')")


def needs(plugin=None, module=None):
    if plugin and not hasattr(core, plugin):
        pytest.skip(f"the {plugin} plugin is not installed")
    if module and importlib.util.find_spec(module) is None:
        pytest.skip(f"{module} is not installed")


# One template per kind of dependency: a plugin wheel, a vsjetpack wrapper, a
# Hybrid script, a DLL from the bundled plugins.7z and a pifroggi package.
@pytest.mark.parametrize("template, plugin, module", [
    ("CAS Sharpen", "cas", None),
    ("Deband", None, "vsdeband"),
    ("MC_Degrain (Simple)", "mv", "vsdenoise"),
    ("SMDegrain", None, "smdegrain"),
    ("Edge Mask", "tcanny", None),
])
def test_representative_templates_run_on_the_fixture(clip, template, plugin, module):
    needs(plugin, module)
    out, ctx = run(clip, template=template)
    assert ctx.notes["template"] == template
    assert out.format.id == clip.format.id
    out.get_frame(out.num_frames // 2)


def test_a_colour_fix_reads_the_reference_input(clip):
    needs(module="vs_colorfix")
    reference = core.std.Invert(clip)
    fixed, _ = run(clip, template="Average Color Fix", reference=reference)
    unchanged, _ = run(clip, template="Average Color Fix")
    stats = lambda c: core.std.PlaneStats(c[4]).get_frame(0).props["PlaneStatsAverage"]  # noqa: E731
    assert abs(stats(fixed) - stats(reference)) < abs(stats(unchanged) - stats(reference))
