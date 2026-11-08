"""
Getting pictures in and out of VapourSynth: loading files, piping a clip
through ffmpeg, and reading frames as arrays.
"""

import shutil
import subprocess
import threading

import numpy as np
import vapoursynth as vs

core = vs.core

# Frame properties that nodes further down a chain read, the same on every
# frame of a clip, kept through an ffmpeg round trip: Telecine's cadence,
# for Bad Telecine Fix.
CARRIED_PROPS = ("TelecineFrames", "TelecineFilm", "TelecinePhase", "TelecineTFF")


def ffmpeg_binary():
    binary = shutil.which("ffmpeg")
    if binary is None:
        raise RuntimeError("ffmpeg was not found on PATH")
    return binary


def load(path):
    """
    Opens a video with BestSource. Its index cache is kept out of the way:
    a dataset run opens many short temporary files whose indexes are never
    needed again.
    """
    return core.bs.VideoSource(source=path, cachemode=0)


def round_trip(clip, out_path, output_args):
    """
    Encodes `clip` to `out_path` with ffmpeg and returns the file loaded back,
    converted to the clip's own format and frame count.

    The clip is sent as y4m, so it must be YUV; RGB is converted with BT.709
    first. ffmpeg is told never to retime: a dropped or doubled frame would
    shift every later LR frame against its HR partner. The CARRIED_PROPS
    of the clip are set on the decoded clip.
    """
    original = clip
    first = clip.get_frame(0).props
    carried = {name: first[name] for name in CARRIED_PROPS if name in first}
    if clip.format.color_family == vs.RGB:
        clip = core.resize.Bicubic(clip, format=vs.YUV444P16 if clip.format.bits_per_sample > 8 else vs.YUV444P8,
                                   matrix_s="709")
    elif clip.format.color_family == vs.GRAY:
        clip = core.resize.Point(clip, format=vs.YUV444P8)

    command = [ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-nostats", "-y",
               "-f", "yuv4mpegpipe", "-i", "-", "-fps_mode", "passthrough", *output_args, out_path]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    errors = []
    reader = threading.Thread(target=lambda: errors.append(process.stderr.read()), daemon=True)
    reader.start()
    try:
        clip.output(process.stdin, y4m=True)
    except (BrokenPipeError, OSError, vs.Error) as e:
        # ffmpeg exiting early closes the pipe; its own message says why.
        process.stdin = None
        process.wait()
        reader.join()
        detail = (errors[0] or b"").decode("utf-8", "replace").strip() if errors else ""
        raise RuntimeError(f"ffmpeg failed: {detail or e}") from e
    process.stdin.close()
    code = process.wait()
    reader.join()
    if code != 0:
        detail = (errors[0] or b"").decode("utf-8", "replace").strip() if errors else ""
        raise RuntimeError(f"ffmpeg failed ({code}): {detail}")

    decoded = load(out_path)
    decoded = convert_like(match_length(decoded, original.num_frames), original)
    return core.std.SetFrameProps(decoded, **carried) if carried else decoded


def match_length(clip, num_frames):
    """Trims or pads (repeating the last frame) to exactly num_frames."""
    if clip.num_frames > num_frames:
        return clip[:num_frames]
    if clip.num_frames < num_frames:
        return clip + clip[-1] * (num_frames - clip.num_frames)
    return clip


def convert_like(clip, like):
    """Converts clip to the format of `like`, keeping its own size."""
    if clip.format.id == like.format.id:
        return clip
    kwargs = {}
    if like.format.color_family == vs.RGB and clip.format.color_family == vs.YUV:
        kwargs["matrix_in_s"] = "709"
    elif like.format.color_family == vs.YUV and clip.format.color_family == vs.RGB:
        kwargs["matrix_s"] = "709"
    return core.resize.Bicubic(clip, format=like.format.id, **kwargs)


def to_rgb24(clip):
    if clip.format.id == vs.RGB24:
        return clip
    kwargs = {"matrix_in_s": "709"} if clip.format.color_family == vs.YUV else {}
    return core.resize.Bicubic(clip, format=vs.RGB24, dither_type="error_diffusion", **kwargs)


def to_gray8(clip):
    if clip.format.color_family == vs.RGB:
        clip = core.resize.Bicubic(clip, format=vs.GRAY8, matrix_s="709")
    elif clip.format.id != vs.GRAY8:
        clip = core.std.ShufflePlanes(clip, 0, vs.GRAY)
        if clip.format.bits_per_sample != 8 or clip.format.sample_type != vs.INTEGER:
            clip = core.resize.Point(clip, format=vs.GRAY8)
    return clip


def frame_array(frame):
    """A frame as an HxWxC uint8 array (planes stacked in clip order)."""
    planes = [np.asarray(frame[i]) for i in range(frame.format.num_planes)]
    return np.stack(planes, axis=-1) if len(planes) > 1 else planes[0]
