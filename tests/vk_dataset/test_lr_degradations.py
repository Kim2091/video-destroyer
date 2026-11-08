"""
Tests for the LR degradation nodes: seeded draws, settings fixed per chunk
while grain changes per frame, output sizes and formats, telecine and its
fix giving back one frame per source frame, and the codec profiles' real stream properties (read back
with ffprobe). Run from the repo root:

    python -m pytest tests/vk_dataset
"""

import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest
import vapoursynth as vs

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "..", "include"))

from vk_dataset import NODE_TYPES  # noqa: E402
from vk_dataset.media import load  # noqa: E402
from vk_dataset.nodes.optics import blur_kernel  # noqa: E402
from vk_dataset.params import sample  # noqa: E402

core = vs.core
FIXTURE = os.path.join(HERE, "fixtures", "two_scenes.mkv")


class FakeContext:
    """What a node function needs of the runner's Context; temp files go to `folder` and are kept."""

    def __init__(self, seed, folder):
        self.rng = random.Random(seed)
        self.notes = {}
        self.folder = folder
        self.temp = []

    def note(self, key, value):
        self.notes[key] = value

    def message(self, text):
        pass

    def temp_path(self, suffix):
        path = os.path.join(self.folder, f"encode_{len(self.temp)}{suffix}")
        self.temp.append(path)
        return path


def apply(name, clip, tmp_path, seed=0, hr=None, **params):
    node_type = NODE_TYPES[name]
    values = {p.name: p.default for p in node_type.params}
    values.update(params)
    ctx = FakeContext(seed, str(tmp_path))
    values = {k: sample(v, ctx.rng) for k, v in values.items()}
    inputs = {"clip": clip} if hr is None else {"clip": clip, "hr": hr}
    return node_type.func(ctx, **inputs, **values)["clip"], ctx


def plane(clip, n, index=0):
    return np.asarray(clip.get_frame(n)[index]).astype(np.float64)


@pytest.fixture(scope="module")
def clip():
    return load(FIXTURE)[:24]


@pytest.fixture(scope="module")
def grey():
    return core.std.BlankClip(width=128, height=96, format=vs.YUV420P8, length=8, color=[126, 128, 128])


SIMPLE = ["optics_blur", "film_grain", "sensor_noise", "resample", "sharpen", "chroma_bleed", "quantize",
          "player_deblock", "player_dnr", "chroma_subsample", "ghost_echo"]


@pytest.mark.parametrize("name", SIMPLE)
def test_seeded_and_shaped(clip, tmp_path, name):
    a, ca = apply(name, clip, tmp_path, seed=7)
    b, cb = apply(name, clip, tmp_path, seed=7)
    assert ca.notes == cb.notes
    assert np.array_equal(plane(a, 5), plane(b, 5))
    assert a.num_frames == clip.num_frames
    others = [apply(name, clip, tmp_path, seed=s)[1].notes for s in range(8, 12)]
    assert any(n != ca.notes for n in others)
    if name != "resample":
        assert (a.width, a.height) == (clip.width, clip.height)
    expected = vs.YUV444P8 if name == "chroma_subsample" else clip.format.id
    assert a.format.id == expected


def test_resample_to_dvd_frames_and_fractions(clip, tmp_path):
    sizes = {(apply("resample", clip, tmp_path, seed=s)[0].width, apply("resample", clip, tmp_path, seed=s)[0].height)
             for s in range(12)}
    assert sizes == {(720, 480), (720, 576)}
    out, ctx = apply("resample", clip, tmp_path, size="scale", scale=0.5, kernel="lanczos", light="linear")
    assert (out.width, out.height) == (160, 120) and ctx.notes["light"] == "linear"
    assert ctx.notes["kernel_settings"]["filter_param_a"] in (2, 3, 4)
    soft, _ = apply("resample", clip, tmp_path, size="scale", scale=0.5, kernel="bilinear", softness=1.6)
    sharp, _ = apply("resample", clip, tmp_path, size="scale", scale=0.5, kernel="bilinear", softness=1.0)
    assert np.abs(np.diff(plane(soft, 3), axis=1)).mean() < np.abs(np.diff(plane(sharp, 3), axis=1)).mean()


def test_scale_to_model_follows_hr(clip, tmp_path):
    sd = core.resize.Bicubic(clip, 720, 480)
    for factor, size in ((1, (320, 240)), (2, (160, 120)), (4, (80, 60))):
        out, _ = apply("scale_to_model", sd, tmp_path, hr=clip, scale=factor)
        assert (out.width, out.height) == size and out.num_frames == clip.num_frames
    odd, _ = apply("scale_to_model", sd, tmp_path, hr=core.resize.Bicubic(clip, 322, 242), scale=2)
    assert (odd.width, odd.height) == (161, 121) and odd.format.id == vs.YUV444P8


def test_blur_kernels():
    for shape in ("gaussian", "generalized", "plateau"):
        kernel = blur_kernel(shape, 1.5, 2.5, 0.6, 1.5)
        assert kernel.shape[0] % 2 == 1 and abs(kernel.sum() - 1) < 1e-4
    round_kernel = blur_kernel("gaussian", 1.5, 1.0, 0.0, 1.0)
    assert np.allclose(round_kernel, round_kernel.T, atol=1e-6)
    wide = blur_kernel("gaussian", 2.0, 3.0, 0.0, 1.0)
    centre = wide.shape[0] // 2
    # Unrotated, the long axis is horizontal.
    assert wide[centre, centre + 2] > wide[centre + 2, centre]


def test_optics_blur_softens(clip, tmp_path):
    out, ctx = apply("optics_blur", clip, tmp_path, shape="gaussian", sigma=1.5, anisotropy=1.0)
    energy = lambda c: (np.diff(plane(c, 4), axis=1) ** 2).mean()  # noqa: E731
    assert energy(out) < 0.5 * energy(clip)
    assert ctx.notes["kernel_size"] == 11


@pytest.mark.parametrize("name", ["film_grain", "sensor_noise"])
def test_grain_is_fresh_per_frame_with_fixed_statistics(grey, tmp_path, name):
    out, _ = apply(name, grey, tmp_path, seed=3, strength=3.0, chroma=0.5)
    residuals = [plane(out, n) - plane(grey, n) for n in range(4)]
    stds = [r.std() for r in residuals]
    assert max(stds) / min(stds) < 1.15 and 1.5 < np.mean(stds) < 6
    assert abs(np.corrcoef(residuals[1].ravel(), residuals[2].ravel())[0, 1]) < 0.1
    assert plane(out, 2, 1).std() > 0.3


def test_grain_size_makes_clumps(grey, tmp_path):
    def neighbour_correlation(size):
        out, _ = apply("film_grain", grey, tmp_path, strength=3.0, size=size)
        r = plane(out, 1) - plane(grey, 1)
        return np.corrcoef(r[:, :-1].ravel(), r[:, 1:].ravel())[0, 1]
    assert neighbour_correlation(1.0) < 0.15 < neighbour_correlation(2.5)


def test_sensor_noise_is_stronger_in_shadows(tmp_path):
    dark = core.std.BlankClip(width=128, height=96, format=vs.YUV420P8, length=2, color=[40, 128, 128])
    bright = core.std.BlankClip(width=128, height=96, format=vs.YUV420P8, length=2, color=[210, 128, 128])
    params = dict(strength=2.0, shot_share=0.7, size=1.0, chroma=0.0)
    dark_noise = (plane(apply("sensor_noise", dark, tmp_path, **params)[0], 1) - 40).std()
    bright_noise = (plane(apply("sensor_noise", bright, tmp_path, **params)[0], 1) - 210).std()
    assert dark_noise > 1.5 * bright_noise


def test_sharpen_halos_luma_only(clip, tmp_path):
    out, ctx = apply("sharpen", clip, tmp_path, radius=1.0, amount=1.0, threshold=0.0, direction="2d")
    assert plane(out, 5).std() > plane(clip, 5).std()
    assert np.array_equal(plane(out, 5, 1), plane(clip, 5, 1))
    cored, _ = apply("sharpen", clip, tmp_path, radius=1.0, amount=1.0, threshold=255.0)
    assert np.array_equal(plane(cored, 5), plane(clip, 5))
    across, _ = apply("sharpen", clip, tmp_path, radius=1.0, amount=1.0, threshold=0.0, direction="horizontal")
    assert not np.array_equal(plane(across, 5), plane(out, 5))


def test_quantize_limits_levels(tmp_path):
    ramp = core.std.BlankClip(width=256, height=16, format=vs.YUV420P8, length=1)
    values = np.tile(np.arange(256, dtype=np.uint8), (16, 1))

    def fill(n, f):
        out = f.copy()
        np.asarray(out[0])[:] = values
        return out
    ramp = core.std.ModifyFrame(ramp, ramp, fill)
    out, _ = apply("quantize", ramp, tmp_path, bits=5, dither="none")
    assert len(np.unique(plane(out, 0))) <= 33
    assert np.abs(plane(out, 0) - values).max() <= 4
    assert out.get_frame(0).props.get("_ColorRange") == ramp.get_frame(0).props.get("_ColorRange")
    dithered, _ = apply("quantize", ramp, tmp_path, bits=5, dither="error diffusion")
    # Dithering keeps the average level of each stretch of the ramp.
    blocks = lambda a: a[:, :240].reshape(16, 15, 16).mean(axis=(0, 2))  # noqa: E731
    assert np.abs(blocks(plane(dithered, 0)) - blocks(values.astype(float))).max() < 1.0


def test_chroma_bleed_moves_colour_not_luma(clip, tmp_path):
    out, _ = apply("chroma_bleed", clip, tmp_path, blur=0.0, shift=2.0)
    assert np.array_equal(plane(out, 3), plane(clip, 3))
    u_out, u_in = plane(out, 3, 1), plane(clip, 3, 1)
    # A 2-pixel luma shift is 1 chroma pixel in 4:2:0.
    assert np.abs(u_out[:, 4:-4] - u_in[:, 3:-5]).mean() < np.abs(u_out[:, 4:-4] - u_in[:, 4:-4]).mean()


def test_chroma_subsample_upsampling(clip, tmp_path):
    blank = core.std.BlankClip(width=64, height=64, format=vs.YUV420P8, length=1)
    noise = np.random.default_rng(0).integers(16, 240, (32, 32), dtype=np.uint8)

    def fill(n, f):
        out = f.copy()
        np.asarray(out[1])[:] = noise
        return out
    busy = core.std.ModifyFrame(blank, blank, fill)
    point, ctx = apply("chroma_subsample", busy, tmp_path, up="point", wrong_siting=0.0)
    assert ctx.notes["up"] == "point" and "down" not in ctx.notes
    smooth, _ = apply("chroma_subsample", busy, tmp_path, up="bilinear", wrong_siting=0.0)
    # Point upsampling repeats each chroma pixel, so half the neighbours match.
    repeats = lambda c: (np.diff(plane(c, 0, 1), axis=1) == 0).mean()  # noqa: E731
    assert repeats(point) > 0.45 > 0.1 > repeats(smooth)
    full = core.resize.Bicubic(clip, format=vs.YUV444P8)
    subsampled, ctx = apply("chroma_subsample", full, tmp_path, format="4:1:1", up="bilinear")
    assert ctx.notes["down"] in ("bicubic", "bilinear", "box") and subsampled.format.id == vs.YUV444P8
    interlaced, ctx = apply("chroma_subsample", clip, tmp_path, interlaced_error=1.0, up="bilinear",
                            wrong_siting=0.0)
    progressive, _ = apply("chroma_subsample", clip, tmp_path, interlaced_error=0.0, up="bilinear",
                           wrong_siting=0.0)
    assert ctx.notes["interlaced_error"] is True
    assert not np.array_equal(plane(interlaced, 3, 1), plane(progressive, 3, 1))


def test_ghost_echo_keeps_brightness(clip, tmp_path):
    out, ctx = apply("ghost_echo", clip, tmp_path, strength=0.15, delay=12.0, blur=0.5, polarity="positive")
    assert ctx.notes["strength"] == 0.15
    assert abs(plane(out, 6).mean() - plane(clip, 6).mean()) < 1.0
    assert not np.array_equal(plane(out, 6), plane(clip, 6))


@pytest.fixture(scope="module")
def moving():
    """A clip whose every frame differs: the fixture shifted one more pixel each frame."""
    base = load(FIXTURE)[30]
    return core.std.Splice([core.resize.Bicubic(base, src_left=n) for n in range(16)])


def telecined(clip, tmp_path, seed=2, source="film", **fix):
    """The clip through Telecine and Bad Telecine Fix, the pulldown between them, and the fix's notes."""
    pulldown, _ = apply("telecine", clip, tmp_path, seed=seed, source=source)
    fixed, ctx = apply("telecine_fix", pulldown, tmp_path, seed=seed, **fix)
    return fixed, pulldown, ctx


def test_telecine_makes_five_frames_of_four(moving, tmp_path):
    film = core.std.AssumeFPS(moving, fpsnum=24000, fpsden=1001)
    out, ctx = apply("telecine", film, tmp_path, seed=2)
    assert out.num_frames == 20 and (out.fps.numerator, out.fps.denominator) == (30000, 1001)
    mixed = [n for n in range(20) if not any(np.array_equal(plane(out, n), plane(film, m)) for m in range(16))]
    assert len(mixed) == 8 and ctx.notes["frames"] == 20
    assert apply("telecine", film, tmp_path, seed=2, source="video")[0].num_frames == 16


def test_telecine_fix_combs_one_frame_in_four(moving, tmp_path):
    out, _, ctx = telecined(moving, tmp_path, recovery="combed", mixed_frames=1.0)
    assert out.num_frames == moving.num_frames and "TelecineFrames" not in out.get_frame(0).props
    changed = [n for n in range(16) if not np.array_equal(plane(out, n), plane(moving, n))]
    assert len(changed) == 4 and len({n % 4 for n in changed}) == 1
    n = changed[1]
    partner = n + (1 if ctx.notes["partner"] == "next" else -1)
    rows = plane(out, n)
    own, other = plane(moving, n), plane(moving, partner)
    first, second = (0, 1) if np.array_equal(rows[0::2], own[0::2]) else (1, 0)
    assert np.array_equal(rows[first::2], own[first::2]) and np.array_equal(rows[second::2], other[second::2])
    clean, _, _ = telecined(moving, tmp_path, recovery="combed", mixed_frames=0.0)
    assert all(np.array_equal(plane(clean, n), plane(moving, n)) for n in range(16))


@pytest.mark.parametrize("recovery", ["blend", "bob", "adaptive"])
def test_telecine_fix_recoveries_keep_frames(moving, tmp_path, recovery):
    out, _, _ = telecined(moving, tmp_path, recovery=recovery)
    assert out.num_frames == moving.num_frames and out.format.id == moving.format.id
    for n in range(16):
        out.get_frame(n)
    # Every recovery loses vertical detail.
    detail = lambda c: np.abs(np.diff(plane(c, 5), axis=0)).mean()  # noqa: E731
    assert detail(out) < detail(moving)


def test_telecine_video_mixes_every_frame(moving, tmp_path):
    out, _, _ = telecined(moving, tmp_path, source="video", recovery="combed")
    assert all(not np.array_equal(plane(out, n), plane(moving, n)) for n in range(15))


def test_telecine_cadence_survives_compression(moving, tmp_path):
    film = core.std.AssumeFPS(core.resize.Bicubic(moving, 720, 480), fpsnum=24000, fpsden=1001)
    pulldown, _ = apply("telecine", film, tmp_path, seed=2)
    encoded, _ = apply("codec", pulldown, tmp_path, profile="dvd_mpeg2", field_coding="always")
    assert encoded.get_frame(0).props["TelecineFrames"] == 16
    out, _ = apply("telecine_fix", encoded, tmp_path, recovery="combed", mixed_frames=0.0)
    assert out.num_frames == 16 and (out.fps.numerator, out.fps.denominator) == (24000, 1001)
    # A clean match of the encoded fields is closest to its own film frame.
    for n in range(16):
        errors = [np.abs(plane(out, n) - plane(film, m)).mean() for m in range(16)]
        assert int(np.argmin(errors)) == n


def test_telecine_fix_skips_untelecined_chunks(moving, tmp_path):
    from vk_dataset.runner import NodeSkipped
    with pytest.raises(NodeSkipped):
        apply("telecine_fix", moving, tmp_path)


def ffprobe(path):
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "frame=pict_type,interlaced_frame,top_field_first:stream=codec_name:format=duration,size",
               "-of", "json", path]
    return json.loads(subprocess.run(command, capture_output=True, text=True, check=True,
                                     stdin=subprocess.DEVNULL).stdout)


@pytest.fixture(scope="module")
def sd_clip():
    """Two seconds of busy 720x480 video, so the encoders spend their bits."""
    base = core.resize.Bicubic(load(FIXTURE)[:48], 720, 480)
    grain = NODE_TYPES["film_grain"].func
    return grain(FakeContext(1, "."), base, strength=6.0, size=1.0, chroma=0.5)["clip"]


def stream_facts(path):
    info = ffprobe(path)
    frames = info["frames"]
    types = "".join(f["pict_type"] for f in frames)
    keyframes = [i for i, t in enumerate(types) if t == "I"]
    seconds = float(info["format"]["duration"])
    kbps = int(info["format"]["size"]) * 8 / seconds / 1000
    return {"codec": info["streams"][0]["codec_name"], "types": types, "keyframes": keyframes, "kbps": kbps,
            "interlaced": all(f.get("interlaced_frame") == 1 for f in frames),
            "top_first": all(f.get("top_field_first") == 1 for f in frames)}


def test_dvd_mpeg2_stream(sd_clip, tmp_path):
    out, ctx = apply("codec", sd_clip, tmp_path, seed=4, profile="dvd_mpeg2", quality=0.5, field_coding="always")
    assert out.num_frames == sd_clip.num_frames and out.format.id == sd_clip.format.id
    facts, notes = stream_facts(ctx.temp[0]), ctx.notes
    assert facts["codec"] == "mpeg2video" and notes["gop"] in (12, 15, 18)
    assert facts["keyframes"][:2] == [0, notes["gop"]]
    assert "BB" in facts["types"] and "BBB" not in facts["types"]
    assert facts["interlaced"] and facts["top_first"]
    # 0.6 bits per pixel at 720x480, 24 fps is about 5 Mbit/s; VBR may spend less on easy frames.
    assert notes["bpp"] == 0.6 and 2000 < facts["kbps"] < 9800


def test_broadcast_mpeg2_is_constant_rate(sd_clip, tmp_path):
    out, ctx = apply("codec", sd_clip, tmp_path, seed=4, profile="broadcast_mpeg2", quality=0.0,
                     field_coding="never")
    facts = stream_facts(ctx.temp[0])
    target = ctx.notes["kbps"]
    assert not facts["interlaced"] and ctx.notes["gop"] in (12, 15)
    assert 0.85 * target < facts["kbps"] < 1.15 * target


def test_web_h264_stream(sd_clip, tmp_path):
    out, ctx = apply("codec", sd_clip, tmp_path, seed=4, profile="web_h264", quality=0.2)
    facts, notes = stream_facts(ctx.temp[0]), ctx.notes
    assert facts["codec"] == "h264" and notes["preset"] in ("ultrafast", "superfast", "veryfast", "faster",
                                                             "fast", "medium", "slow")
    assert notes["crf"] == 28.0 and facts["kbps"] < notes["peak_kbps"] * 1.3
    if notes["preset"] != "ultrafast":
        assert "B" in facts["types"]


def test_web_h265_and_xvid_streams(sd_clip, tmp_path):
    _, ctx = apply("codec", sd_clip, tmp_path, profile="web_h265", quality=0.5)
    assert stream_facts(ctx.temp[0])["codec"] == "hevc" and ctx.notes["crf"] == 28.0
    _, ctx = apply("codec", sd_clip, tmp_path, profile="xvid", quality=0.0)
    facts = stream_facts(ctx.temp[0])
    assert facts["codec"] == "mpeg4" and "B" in facts["types"] and ctx.notes["quantizer"] == 8
