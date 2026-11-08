"""
Tests for the analogue video nodes (Composite Encode, VHS Tape, Playback
Deck, Composite Decode) and their VapourSynth plugin (native/vs-ntsc). They
need the plugin built into include/vk_dataset/plugins (npm run build:native)
and are skipped without it.

The CLI identity tests also need ffmpeg on the PATH and the simulator's
command-line tool, named by the NTSC_SIMULATOR_CLI environment variable or
built in native/ntsc-simulator with

    cargo build --release --locked --no-default-features --bin ntsc-composite-simulator
        --target-dir ../vs-ntsc/build/cli

and are skipped without them.
"""

import os
import random
import shutil
import subprocess
import sys

import numpy as np
import pytest
import vapoursynth as vs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "include"))

from vk_dataset import NODE_TYPES, node_specs  # noqa: E402
from vk_dataset.media import load  # noqa: E402
from vk_dataset.nodes.ntsc_chain import DECKS, DEVICE_PARAMS, PLUGIN, ntsc, parse_more  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(PLUGIN), reason="plugin not built (npm run build:native)")

core = vs.core
HERE = os.path.dirname(__file__)
FIXTURE = os.path.join(HERE, "fixtures", "two_scenes.mkv")
CLI = os.environ.get("NTSC_SIMULATOR_CLI") or os.path.join(
    HERE, "..", "..", "native", "vs-ntsc", "build", "cli", "release",
    "ntsc-composite-simulator.exe" if sys.platform == "win32" else "ntsc-composite-simulator")


class FakeContext:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.notes = {}

    def note(self, key, value):
        self.notes[key] = value


def call(name, seed=0, **params):
    """Runs a node with its defaults and `params`; inputs are the params its type takes as inputs."""
    node_type = NODE_TYPES[name]
    values = {p.name: p.default for p in node_type.params}
    inputs = {k: params.pop(k) for k in list(params) if k in node_type.inputs}
    values.update(params)
    ctx = FakeContext(seed)
    out = node_type.func(ctx, **inputs, **values)
    return next(iter(out.values())), ctx.notes


def encode(clip, seed=0, **params):
    return call("ntsc_encode", seed, clip=clip, **params)[0]


def tape(signal, seed=1, **params):
    return call("ntsc_tape", seed, signal=signal, **params)[0]


def deck(signal, **params):
    return call("ntsc_deck", signal=signal, **params)[0]


def decode(signal, **params):
    return call("ntsc_decode", signal=signal, **params)[0]


def frames(clip, order=None):
    order = range(clip.num_frames) if order is None else order
    got = {n: np.dstack([np.asarray(f[p]) for p in range(3)]) for n, f in
           ((n, clip.get_frame(n)) for n in order)}
    return np.stack([got[n] for n in range(clip.num_frames)])


@pytest.fixture(scope="module")
def rgb():
    return core.resize.Bicubic(load(FIXTURE)[:40], format=vs.RGB24, matrix_in_s="709")


# -- the plugin ----------------------------------------------------------------

def test_seeded_and_deterministic(rgb):
    a = frames(ntsc().Simulate(rgb[:6], preset="lq", seed=5))
    b = frames(ntsc().Simulate(rgb[:6], preset="lq", seed=5))
    c = frames(ntsc().Simulate(rgb[:6], preset="lq", seed=6))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert not np.array_equal(a, frames(rgb[:6]))


@pytest.mark.parametrize("preset,deck_name,extra", [
    ("mq", "", {"comb": "3d"}),  # the 3D comb reads the two frames before
    ("hq", "deck-svhs-tbc", {}),  # the recursive noise reduction carries state
])
def test_frame_order_does_not_matter(rgb, preset, deck_name, extra):
    clip = ntsc().Simulate(rgb, preset=preset, deck=deck_name, seed=2, **extra)
    forward = frames(clip)
    clip = ntsc().Simulate(rgb, preset=preset, deck=deck_name, seed=2, **extra)
    order = list(range(rgb.num_frames))
    random.Random(1).shuffle(order)
    assert np.array_equal(forward, frames(clip, order))


@pytest.mark.parametrize("fmt", [vs.RGB24, vs.RGB48, vs.RGBS])
def test_formats_size_and_frame_count(rgb, fmt):
    clip = core.resize.Point(rgb[:4], format=fmt)
    out = ntsc().Simulate(clip, preset="mq", seed=1)
    assert (out.format.id, out.width, out.height, out.num_frames) == (fmt, clip.width, clip.height, 4)
    out.get_frame(3)
    resized = ntsc().Simulate(clip, preset="mq", width=640, height=480)
    assert (resized.width, resized.height) == (640, 480)


def test_rejects_bad_input(rgb):
    with pytest.raises(vs.Error, match="RGB24, RGB48 or RGBS"):
        ntsc().Simulate(core.resize.Bicubic(rgb, format=vs.YUV420P8, matrix_s="709"))
    with pytest.raises(vs.Error, match="unknown preset"):
        ntsc().Simulate(rgb, preset="dvd-hq")
    with pytest.raises(vs.Error, match="tape_snr|tape-snr"):
        ntsc().Simulate(rgb, tape_snr="loud")
    with pytest.raises(vs.Error, match="dubs"):
        ntsc().Simulate(rgb, dubs=-1)


def test_dubs_add_generations(rgb):
    dubbed = frames(ntsc().Simulate(rgb[:4], preset="mq", seed=3, generations="1", dubs=2))
    third = frames(ntsc().Simulate(rgb[:4], preset="mq", seed=3, generations="3"))
    first = frames(ntsc().Simulate(rgb[:4], preset="mq", seed=3, generations="1"))
    assert np.array_equal(dubbed, third)
    assert not np.array_equal(dubbed, first)


def test_every_plugin_setting_belongs_to_one_device():
    taken = {"clip", "preset", "deck", "seed", "dubs", "width", "height"}
    plugin = {part.split(":")[0] for part in ntsc().Simulate.signature.split(";") if part} - taken
    owned = [name.replace("-", "_") for params in DEVICE_PARAMS.values() for name in params]
    assert len(owned) == len(set(owned))
    assert set(owned) == plugin


# -- the nodes -----------------------------------------------------------------

def test_palette_lists_the_devices_together_in_signal_order():
    types = [spec["type"] for spec in node_specs()]
    start = types.index("ntsc_encode")
    assert types[start:start + 4] == ["ntsc_encode", "ntsc_tape", "ntsc_deck", "ntsc_decode"]


def test_chain_keeps_format_size_and_frames_and_notes_its_draws():
    clip = load(FIXTURE)[:8]
    signal, notes = call("ntsc_tape", 3, signal=encode(clip), tape="any")
    assert notes["tape"] in ("lq", "mq", "hq") and notes["tape"] == signal.tape
    assert notes["seed"] == signal.seed
    out = decode(deck(signal, deck="svhs-tbc"))
    assert (out.format.id, out.width, out.height, out.num_frames) == (clip.format.id, clip.width, clip.height, 8)
    again = decode(deck(tape(encode(clip), seed=3), deck="svhs-tbc"))
    other = decode(deck(tape(encode(clip), seed=4), deck="svhs-tbc"))
    assert np.array_equal(np.asarray(out.get_frame(5)[0]), np.asarray(again.get_frame(5)[0]))
    assert not np.array_equal(np.asarray(out.get_frame(5)[0]), np.asarray(other.get_frame(5)[0]))


def test_one_tape_feeds_two_decks(rgb):
    recorded = tape(encode(rgb[:6]), tape="mq")
    before = frames(decode(recorded))
    consumer = frames(decode(deck(recorded, deck="consumer")))
    tbc = frames(decode(deck(recorded, deck="svhs-tbc")))
    # Each deck is its own chain; neither changes the tape's signal.
    assert recorded.deck is None and np.array_equal(before, frames(decode(recorded)))
    assert np.array_equal(before, consumer)
    assert not np.array_equal(consumer, tbc)


def test_dubs_and_their_limits(rgb):
    first = tape(encode(rgb[:4]), tape="mq", generations=1)
    dub = tape(first, seed=7, generations=2)
    assert (dub.tape, dub.dubs, dict(dub.args)["generations"]) == ("mq", 2, "1")
    assert dub.seed != first.seed
    with pytest.raises(ValueError, match="dub"):
        tape(first, tape="lq")
    with pytest.raises(ValueError, match="dub"):
        tape(first, tape_snr=30.0)
    played = deck(first)
    with pytest.raises(ValueError, match="before the Playback Deck"):
        tape(played)
    with pytest.raises(ValueError, match="one Playback Deck"):
        deck(played)


def test_settings_reach_the_plugin_by_device(rgb):
    signal = deck(tape(encode(rgb[:2], snow=0.05, sharpening=0.3), tape="hq", tape_snr=33.5, generations=2,
                       more="rf-loss=0.5"),
                  deck="edit", sharpness=-0.25, output="composite")
    assert dict(signal.args) == {
        "noise": "0.05", "edge_ringing": "0.3", "reception_before_tape": "true", "tape_snr": "33.5",
        "generations": "2", "rf_loss": "0.5", "deck_sharpness": "-0.25", "video_out": "composite"}
    assert (signal.tape, signal.deck) == ("hq", "edit")


def test_more_settings():
    assert parse_more("tracking-bars=0..0.3  generations=2;wobble=1", "VHS Tape") == {
        "tracking_bars": "0..0.3", "generations": "2", "wobble": "1"}
    with pytest.raises(ValueError, match="name=value"):
        parse_more("generations", "VHS Tape")
    with pytest.raises(ValueError, match="deck-nc is a setting of Playback Deck"):
        parse_more("deck-nc=4", "VHS Tape")
    with pytest.raises(ValueError, match="not a setting"):
        parse_more("loudness=4", "VHS Tape")


def test_every_signal_previews(rgb):
    source = encode(rgb[:3])
    recorded = tape(source, tape="lq")
    played = deck(recorded, deck="vhs-decode")
    for signal in (source, recorded, played):
        clip = signal.preview
        assert (clip.width, clip.height, clip.num_frames) == (rgb.width, rgb.height, 3)
        assert signal.preview is clip
    assert np.array_equal(frames(recorded.preview), frames(decode(recorded)))


# -- the CLI -------------------------------------------------------------------

def cli_args(signal, decoder):
    """The simulator CLI's flags for a chain and its decoder settings, its dubs folded into generations."""
    args = {**dict(signal.args), **decoder}
    out = ["--seed", str(signal.seed)]
    if signal.tape:
        out += ["--preset", signal.tape]
    if DECKS.get(signal.deck or "consumer"):
        out += ["--deck", DECKS[signal.deck]]
    if signal.dubs:
        args["generations"] = str(int(args["generations"]) + signal.dubs)
    for name, value in args.items():
        flag = "--" + name.replace("_", "-")
        out += [flag] if value == "true" else [flag, value]
    return out


def rgb24(clip):
    return frames(clip).astype(np.uint8).tobytes()


@pytest.mark.skipif(not os.path.exists(CLI) or shutil.which("ffmpeg") is None,
                    reason="the simulator CLI or ffmpeg is missing")
@pytest.mark.parametrize("chain", ["tape only", "every device and a dub"])
def test_chain_matches_the_cli(tmp_path, rgb, chain):
    # 12 frames, so the deck's noise reduction runs unbroken from frame 0, as on the CLI.
    clip = rgb[:12]
    if chain == "tape only":
        signal, peaking, decoder = tape(encode(clip), seed=11, tape="mq"), 0.0, {}
    else:
        source = encode(clip, snow=0.04, ghost=0.1, sharpening=0.35)
        recorded = tape(source, seed=12, tape="lq", tape_snr=31.0, generations=1, more="tracking-bars=0.4")
        signal = deck(tape(recorded, seed=13), deck="svhs-tbc", sharpness=0.1, output="composite")
        peaking, decoder = 0.25, {"capture_peaking": "0.25", "capture_peaking_mhz": "2.6"}
    ours = rgb24(decode(signal, peaking=peaking))

    source_file, raw, output = (tmp_path / name for name in ("in.mkv", "raw.rgb", "out.mkv"))
    w, h = clip.width, clip.height
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                    "-r", "30000/1001", "-i", "-", "-c:v", "ffv1", "-pix_fmt", "bgr0", str(source_file)],
                   input=rgb24(clip), check=True)
    subprocess.run([CLI, "roundtrip", str(source_file), "-o", str(output), "--width", str(w), "--height", str(h),
                    "--raw-copy", str(raw), *cli_args(signal, decoder)], check=True, capture_output=True)
    # The raw copy is the CLI's output before its lossless writer; the file must decode to the same.
    assert raw.read_bytes() == ours
    written = subprocess.run(["ffmpeg", "-v", "error", "-i", str(output), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             check=True, capture_output=True).stdout
    assert written == ours
