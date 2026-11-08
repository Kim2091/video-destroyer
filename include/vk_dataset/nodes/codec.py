"""
Compression with real encoders, set up the way real releases are encoded.
The whole chunk is one stream, so frame types (I, P, B), rate control and
motion behave as they do on a disc or a website.

Profiles:
  dvd_mpeg2        MPEG-2 VBR within the DVD buffer (VBV) and peak rate,
                   GOP 12/15/18 with 2 B-frames, optionally field coded.
  broadcast_mpeg2  MPEG-2 at a constant rate, GOP 12/15, 2 B-frames.
  web_h264         x264 at the fast presets most web video uses, CRF capped
                   by a VBV peak rate, 2 to 10 s between keyframes.
  web_h265         x265 CRF, fast to slow presets.
  xvid             MPEG-4 ASP at a fixed quantizer with B-frames, as old
                   rips were made (libxvid when ffmpeg has it).

Bitrates are set in bits per pixel per frame (bitrate / (width x height x
fps)), so a profile means the same at any frame size; the DVD and broadcast
peak rate and buffer are scaled from their SD values the same way.
Quality 0 is the worst end of a profile's realistic range, 1 the best.
A second encode, such as a rip of a disc, is a second Compression node.
"""

import functools
import subprocess

import vapoursynth as vs

from ..media import ffmpeg_binary, round_trip
from ..registry import Param, node
from .resample import even

core = vs.core

PROFILES = ["dvd_mpeg2", "broadcast_mpeg2", "web_h264", "web_h265", "xvid"]
FIELD_CODING = ["always", "never"]

# NTSC DVD: 720x480 at 29.97 frames a second, 9.8 Mbit/s peak, 1.8 Mbit buffer.
SD_PIXEL_RATE = 720 * 480 * 30000 / 1001
DVD_PEAK = 9_800_000
DVD_BUFFER = 1_835_008

X264_PRESETS = {"ultrafast": 0.03, "superfast": 0.07, "veryfast": 0.3, "faster": 0.2, "fast": 0.15,
                "medium": 0.2, "slow": 0.05}
X264_TUNES = {"none": 0.6, "film": 0.15, "animation": 0.15, "grain": 0.1}
X265_PRESETS = {"fast": 0.3, "medium": 0.5, "slow": 0.2}


@functools.lru_cache(maxsize=1)
def encoders():
    """The video encoders this ffmpeg was built with."""
    out = subprocess.run([ffmpeg_binary(), "-hide_banner", "-encoders"], capture_output=True, text=True,
                         stdin=subprocess.DEVNULL).stdout
    return {line.split()[1] for line in out.splitlines() if line.startswith(" V")}


def weighted(rng, table):
    names = list(table)
    return rng.choices(names, weights=[table[n] for n in names])[0]


def lerp(low, high, quality):
    return low + (high - low) * quality


def profile_args(profile, rng, quality, width, height, fps, field_coded):
    """The ffmpeg output arguments of one encode, and what was drawn for the manifest."""
    pixel_rate = width * height * fps
    sd_ratio = pixel_rate / SD_PIXEL_RATE
    notes = {"profile": profile, "quality": round(quality, 3)}
    if profile in ("dvd_mpeg2", "broadcast_mpeg2"):
        dvd = profile == "dvd_mpeg2"
        bpp = lerp(0.3, 0.9, quality) if dvd else lerp(0.2, 0.5, quality)
        rate = round(bpp * pixel_rate)
        gop = rng.choice([12, 15, 18] if dvd else [12, 15])
        dc = rng.choice([8, 9, 10])
        args = ["-c:v", "mpeg2video", "-b:v", str(rate), "-bufsize", str(round(DVD_BUFFER * sd_ratio)),
                "-g", str(gop), "-bf", "2", "-dc", str(dc)]
        if dvd:
            args += ["-maxrate", str(max(round(rate * 1.15), round(DVD_PEAK * sd_ratio)))]
        else:
            args += ["-maxrate", str(rate), "-minrate", str(rate)]
        if field_coded:
            args += ["-flags", "+ilme+ildct", "-top", "1", "-alternate_scan", "1"]
        notes.update(bpp=round(bpp, 3), kbps=round(rate / 1000), gop=gop, b_frames=2, dc_precision=dc,
                     field_coded=field_coded)
        return args, notes
    if profile == "web_h264":
        preset, tune = weighted(rng, X264_PRESETS), weighted(rng, X264_TUNES)
        crf = round(lerp(30, 20, quality), 1)
        cap_bpp = lerp(0.08, 0.25, quality)
        peak = round(cap_bpp * pixel_rate)
        gop = max(1, round(fps * rng.uniform(2.0, 10.0)))
        # x264's rate cap gives a different encode on every run when it has more than one thread.
        args = ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-maxrate", str(peak),
                "-bufsize", str(peak * 2), "-g", str(gop), "-profile:v", "high", "-threads", "1"]
        if tune != "none":
            args += ["-tune", tune]
        notes.update(preset=preset, tune=tune, crf=crf, peak_kbps=round(peak / 1000), gop=gop)
        return args, notes
    if profile == "web_h265":
        preset = weighted(rng, X265_PRESETS)
        crf = round(lerp(32, 24, quality), 1)
        gop = max(1, round(fps * rng.uniform(2.0, 10.0)))
        args = ["-c:v", "libx265", "-preset", preset, "-crf", str(crf),
                "-x265-params", f"log-level=error:keyint={gop}"]
        notes.update(preset=preset, crf=crf, gop=gop)
        return args, notes
    if profile == "xvid":
        encoder = "libxvid" if "libxvid" in encoders() else "mpeg4"
        q = round(lerp(8, 3, quality))
        gop = max(1, round(fps * 10))
        args = ["-c:v", encoder, "-q:v", str(q), "-bf", "2", "-g", str(gop)]
        if encoder == "mpeg4":
            args += ["-vtag", "XVID"]
        notes.update(encoder=encoder, quantizer=q, gop=gop, b_frames=2)
        return args, notes
    raise ValueError(f"unknown codec profile {profile!r}; known: {', '.join(PROFILES)}")


def encode(ctx, clip, profile, quality, field_coded):
    fps = clip.fps.numerator / clip.fps.denominator if clip.fps.denominator else 24.0
    args, notes = profile_args(profile, ctx.rng, quality, clip.width, clip.height, fps, field_coded)
    args += ["-pix_fmt", "yuv420p", "-colorspace", "bt709"]
    return round_trip(clip, ctx.temp_path(".mkv"), args), notes


@node("codec", category="Degrade", label="Compression", probability=True,
      inputs={"clip": "clip"}, outputs={"clip": "clip"},
      params=[
          Param("profile", "choice", "web_h264", choices=PROFILES, help="Which kind of release to imitate."),
          Param("quality", "float", {"uniform": [0.0, 1.0]}, random=True,
                help="0 is the worst of the profile's realistic range, 1 the best.", min=0.0, max=1.0),
          Param("field_coding", "choice", {"choice": FIELD_CODING}, choices=FIELD_CODING,
                help="MPEG-2: encode as interlaced fields, as most DVDs of telecined video are."),
      ])
def codec(ctx, clip, *, profile, quality, field_coding):
    if field_coding not in FIELD_CODING:
        raise ValueError(f"field coding must be one of {', '.join(FIELD_CODING)}")
    field_coded = profile.endswith("mpeg2") and field_coding == "always"
    # Every profile here encodes 4:2:0, which needs an even size.
    if clip.width % 2 or clip.height % 2:
        clip = core.resize.Bicubic(clip, even(clip.width), even(clip.height))
    clip, notes = encode(ctx, clip, profile, quality, field_coded)
    for key, value in notes.items():
        ctx.note(key, value)
    return {"clip": clip}
