"""
Processing HR targets: line darkening, dehaloing, tone adjustments and gamma
matching against the LR side, then writing the result beside the originals.

The line and halo filters are vs-jetpack's own, the same calls the app's Fast
Line Darken and Fine Dehalo filters make. Both act on luma, so each runs on
the Rec.709 luma of the RGB frame and its change is added back to R, G and B,
which leaves the colour of the picture alone.

Frames are kept as float RGB between steps so a chain of steps rounds to 8-bit
once, in the writer. HR clips carry their file stem in the VKName frame prop
and an index in VKOwner (see dataset_source.py); the writer names files by it.
"""

import json
import math
import os

import numpy as np
import vapoursynth as vs

from .. import match_gamma
from ..media import frame_array, to_rgb24
from ..registry import Param, node

core = vs.core


def rgb_float(clip):
    """The clip as 32-bit float RGB, which is what every step here works in."""
    if clip.format.id == vs.RGBS:
        return clip
    if clip.format.color_family == vs.RGB:
        return core.resize.Point(clip, format=vs.RGBS)
    return core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")


def on_luma(clip, func):
    """Runs func on the clip's 16-bit full-range luma and adds the change it makes back to R, G and B."""
    rgb = rgb_float(clip)
    luma = core.resize.Bilinear(rgb, format=vs.GRAY16, matrix_s="709", range_s="full")
    changed = func(luma)
    delta = core.std.Expr([changed, luma], "x y - 65535 /", format=vs.GRAYS)
    delta = core.std.ShufflePlanes([delta] * 3, [0, 0, 0], vs.RGB)
    return core.std.Expr([rgb, delta], "x y + 0 max 1 min")


@node("fast_line_darken", category="HR Process", label="Fast Line Darken",
      description="Darkens line art with vsrgtools' fast_line_darken, on luma.",
      inputs={"clip": "clip"}, outputs={"clip": "clip"}, probability=True,
      params=[
          Param("strength", "float", 48.0, help="Line darkening amount.", min=0.0),
          Param("protection", "float", 5.0, help="Protects the darkest lines.", min=0.0),
          Param("luma_cap", "float", 191.0, help="Luma on the 0-255 scale above which nothing is darkened.",
                min=0.0, max=255.0),
          Param("threshold", "float", 4.0, help="Smallest difference, on the 0-255 scale, that counts as a line.",
                min=0.0, max=255.0),
          Param("thinning", "float", 0.0, help="Thins lines instead of only darkening them (0 = off).", min=0.0),
      ])
def fast_line_darken(ctx, clip, *, strength, protection, luma_cap, threshold, thinning):
    from vsrgtools import fast_line_darken as darken
    return {"clip": on_luma(clip, lambda y: darken(y, strength=strength, protection=protection,
                                                    luma_cap=luma_cap, threshold=threshold, thinning=thinning))}


@node("fine_dehalo", category="HR Process", label="Fine Dehalo",
      description="Removes bright and dark halos with vsdehalo's fine_dehalo, on luma.",
      inputs={"clip": "clip"}, outputs={"clip": "clip"}, probability=True,
      params=[
          Param("blur", "float", 1.4, help="Gaussian sigma of the dehalo blur.", min=0.0),
          Param("lowsens", "float", 50.0, help="Dehalo is applied in full below this.", min=0.0, max=100.0),
          Param("highsens", "float", 50.0, help="Dehalo is skipped above this.", min=0.0, max=100.0),
          Param("ss", "float", 1.5, help="Supersampling factor (1 = off).", min=1.0),
          Param("darkstr", "float", 0.0, help="Dark halo strength.", min=0.0),
          Param("brightstr", "float", 1.0, help="Bright halo strength.", min=0.0),
          Param("rx", "int", 2, help="Halo removal radius, horizontal.", min=1),
          Param("ry", "int", 0, help="Halo removal radius, vertical (0 = same as rx).", min=0),
          Param("thmi", "float", 80.0, help="Strong edge ramp, low end.", min=0.0, max=255.0),
          Param("thma", "float", 128.0, help="Strong edge ramp, high end.", min=0.0, max=255.0),
          Param("thlimi", "float", 50.0, help="Weak edge ramp, low end.", min=0.0, max=255.0),
          Param("thlima", "float", 100.0, help="Weak edge ramp, high end.", min=0.0, max=255.0),
          Param("exclude", "bool", True, help="Protects detail close to strong edges."),
          Param("exclude_strength", "float", 1.0, min=0.0, max=1.0,
                help="How much of that protection applies: 1 is full, 0 is the same as exclude off."),
          Param("edgeproc", "float", 0.0, help="Blends raw edges back into the mask (0 = off).", min=0.0),
          Param("contra", "float", 0.0, help="Contra-sharpening after the dehalo (0 = off).", min=0.0),
          Param("pre_ss", "int", 1, help="Upscale factor before dehaloing (1 = off).", min=1, max=4),
      ])
def fine_dehalo(ctx, clip, *, blur, lowsens, highsens, ss, darkstr, brightstr, rx, ry, thmi, thma, thlimi,
                thlima, exclude, exclude_strength, edgeproc, contra, pre_ss):
    from vsdehalo import fine_dehalo as dehalo

    def run(luma, excluded):
        return dehalo(luma, blur=blur, lowsens=lowsens, highsens=highsens, ss=ss, darkstr=darkstr,
                      brightstr=brightstr, rx=rx, ry=ry or None, thmi=thmi, thma=thma, thlimi=thlimi,
                      thlima=thlima, exclude=excluded, edgeproc=edgeproc, contra=contra, pre_ss=pre_ss)

    # The result is the source plus the mask times the dehalo's change, so a
    # partial exclusion is the blend of the runs with it off and on.
    strength = float(exclude_strength) if exclude else 0.0

    def func(luma):
        if strength <= 0.0:
            return run(luma, False)
        if strength >= 1.0:
            return run(luma, True)
        return core.std.Merge(run(luma, False), run(luma, True), weight=strength)

    return {"clip": on_luma(clip, func)}


@node("tone_adjust", category="HR Process", label="Brightness / Contrast / Gamma",
      description="Fixed brightness, contrast (pivoting at mid grey) and gamma on RGB, clipped to 0-1.",
      inputs={"clip": "clip"}, outputs={"clip": "clip"}, probability=True,
      params=[
          Param("brightness", "float", 0.0, help="Added to every channel.", min=-1.0, max=1.0),
          Param("contrast", "float", 0.0, help="Contrast around mid grey.", min=-1.0, max=1.0),
          Param("gamma", "float", 1.0, help="Power curve; above 1 brightens shadows.", min=0.1),
      ])
def tone_adjust(ctx, clip, *, brightness, contrast, gamma):
    rgb = rgb_float(clip)
    if contrast == 0.0 and brightness == 0.0 and gamma == 1.0:
        return {"clip": rgb}
    slope = math.tan((min(max(contrast, -0.99), 0.99) + 1.0) * math.pi / 4.0)
    expr = f"x 0.5 - {slope!r} * 0.5 + {float(brightness)!r} + 0 max 1 min"
    if gamma != 1.0:
        expr += f" {1.0 / gamma!r} pow"
    return {"clip": core.std.Expr(rgb, expr)}


def _copy_with(frame, planes):
    out = frame.copy()
    for index, plane in enumerate(planes):
        np.copyto(np.asarray(out[index]), plane)
    return out


@node("match_gamma", category="HR Process", label="Match Gamma",
      description="Estimates the gamma that makes the HR image's tones match the LR image's, from matched "
                  "quantiles of both, and applies it to HR ('hr_to_lr') or to the LR frames ('lr_to_hr').",
      inputs={"hr": "clip", "lr": "clip", "lr_all": "clip"}, optional_inputs=("lr_all",),
      outputs={"hr": "clip", "lr": "clip", "lr_all": "clip"}, probability=True,
      params=[
          Param("direction", "choice", "hr_to_lr", choices=["hr_to_lr", "lr_to_hr"],
                help="Which side is adjusted. 'lr_to_hr' changes the LR clips and leaves HR as it is."),
          Param("scope", "choice", "scene", choices=["scene", "show"],
                help="'scene' fits one gamma per HR/LR pair; 'show' pools samples of every pair into one gamma."),
          Param("min_gamma", "float", 0.2, min=0.01, help="Lowest gamma the estimate may take."),
          Param("max_gamma", "float", 5.0, min=0.01, help="Highest gamma the estimate may take."),
          Param("max_samples", "int", 1_000_000, min=0, help="Pixels sampled per image for the fit (0 = all)."),
      ])
def match_gamma_node(ctx, hr, lr, lr_all=None, *, direction, scope, min_gamma, max_gamma, max_samples):
    hr, lr = rgb_float(hr), rgb_float(lr)
    low, high = sorted((float(min_gamma), float(max_gamma)))
    to_lr = direction == "hr_to_lr"
    scene_gammas = {}
    ctx.note("scene_gammas", scene_gammas)

    show_gamma = None
    if scope == "show":
        count = min(hr.num_frames, lr.num_frames)
        budget = max(4096, max_samples // max(1, count)) if max_samples else 0
        images, references = [], []
        for n in range(count):
            hr_log = match_gamma.collect_log_samples(frame_array(hr.get_frame(n)), budget)
            lr_log = match_gamma.collect_log_samples(frame_array(lr.get_frame(n)), budget)
            images.append(hr_log if to_lr else lr_log)
            references.append(lr_log if to_lr else hr_log)
        pooled = match_gamma.estimate_from_logs(np.concatenate(images), np.concatenate(references))
        show_gamma = min(max(pooled, low), high)
        ctx.note("show_gamma", round(show_gamma, 6))
        ctx.message(f"Match Gamma: one gamma of {show_gamma:.4f} for {count} pairs")

    def gamma_of(owner, hr_frame, lr_frame):
        if show_gamma is not None:
            return show_gamma
        name = hr_frame.props.get("VKName", str(owner))
        if name not in scene_gammas:
            hr_image, lr_image = frame_array(hr_frame), frame_array(lr_frame)
            image, reference = (hr_image, lr_image) if to_lr else (lr_image, hr_image)
            gamma = match_gamma.estimate(image, reference, max_samples)
            scene_gammas[name] = round(min(max(gamma, low), high), 6)
        return scene_gammas[name]

    def adjusted(frame, gamma):
        image = match_gamma.apply(frame_array(frame), gamma)
        return _copy_with(frame, [image[:, :, p] for p in range(3)])

    def owner_of(n, frame):
        return int(frame.props.get("VKOwner", n))

    out = {"hr": hr, "lr": lr}
    if to_lr:
        def select_hr(n, f):
            return adjusted(f[0], gamma_of(owner_of(n, f[0]), f[0], f[1]))
        out["hr"] = core.std.ModifyFrame(hr, [hr, lr], select_hr)
        if lr_all is not None:
            out["lr_all"] = rgb_float(lr_all)
        return out

    def select_lr(n, f):
        return adjusted(f[1], gamma_of(owner_of(n, f[0]), f[0], f[1]))
    out["lr"] = core.std.ModifyFrame(lr, [hr, lr], select_lr)
    if lr_all is not None:
        lr_all = rgb_float(lr_all)
        total = lr_all.num_frames

        def owned(clip):
            # Frame n of the result is the pair frame that owns LR frame n.
            return core.std.FrameEval(clip[0] * total, lambda n, f: clip[int(f.props["VKOwner"])] * total,
                                      prop_src=[lr_all])

        def select_all(n, f):
            return adjusted(f[2], gamma_of(owner_of(n, f[2]), f[0], f[1]))
        out["lr_all"] = core.std.ModifyFrame(lr_all, [owned(hr), owned(lr), lr_all], select_all)
    return out


def _marker(directory, payload):
    with open(os.path.join(directory, ".incomplete.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _to_rgb24(clip):
    """Rounds float RGB to 8 bits; the dithering of media.to_rgb24 would add noise to a cleaned picture."""
    if clip.format.color_family == vs.RGB:
        return core.resize.Point(clip, format=vs.RGB24)
    return to_rgb24(clip)


def _write_frames(ctx, clip, folder, label, done, total):
    from .output import write_image
    names = []
    for n, frame in enumerate(_to_rgb24(clip).frames(close=True)):
        name = frame.props.get("VKName")
        if name is None:
            raise ValueError(f"the {label} clip has no VKName frame prop; build it from a Load Dataset node")
        name = f"{name.decode() if isinstance(name, bytes) else name}.png"
        write_image(os.path.join(folder, name), frame_array(frame), "png", 0)
        names.append(name)
        if (n + 1) % 25 == 0:
            ctx.emit({"progress": round((done + n + 1) / total, 4),
                      "message": f"writing {label}: {n + 1} of {clip.num_frames}"})
    return names


@node("write_processed", category="Output", label="Write Processed Dataset",
      description="Writes processed HR frames to hr_p/ and, when LR frames are connected, adjusted LR frames "
                  "to lr_p/, both beside hr/ and lr/ in the dataset folder, as 8-bit PNG named like the "
                  "originals. The originals are not touched.",
      inputs={"hr": "clip", "lr_all": "clip"}, optional_inputs=("lr_all",),
      params=[
          Param("directory", "path", "", help="The dataset folder that holds hr/ and lr/."),
          Param("hr_folder", "str", "hr_p", help="Folder for the processed HR images."),
          Param("lr_folder", "str", "lr_p", help="Folder for the adjusted LR images."),
      ])
def write_processed(ctx, hr, lr_all=None, *, directory, hr_folder, lr_folder):
    if not directory or not os.path.isdir(directory):
        raise FileNotFoundError(f"the dataset folder {directory!r} does not exist")
    outputs = {"hr": hr_folder}
    if lr_all is not None:
        outputs["lr"] = lr_folder
    for side, folder in outputs.items():
        if not folder or os.path.basename(folder) != folder or folder.lower() in ("hr", "lr"):
            raise ValueError(f"the {side} output folder {folder!r} must be a new folder name beside hr/ and lr/")
    if len(set(outputs.values())) != len(outputs):
        raise ValueError("the HR and LR output folders must differ")
    folders = {side: os.path.join(directory, folder) for side, folder in outputs.items()}
    payload = {"status": "running", "outputs": folders}
    for path in folders.values():
        os.makedirs(path, exist_ok=True)
        _marker(path, payload)

    total = hr.num_frames + (lr_all.num_frames if lr_all is not None else 0)
    written = {}
    try:
        written["hr"] = _write_frames(ctx, hr, folders["hr"], "HR", 0, total)
        if lr_all is not None:
            written["lr"] = _write_frames(ctx, lr_all, folders["lr"], "LR", hr.num_frames, total)
    except Exception as e:
        for path in folders.values():
            _marker(path, {**payload, "status": "failed", "error": str(e),
                           "files_written": {k: len(v) for k, v in written.items()}})
        raise
    for path in folders.values():
        os.remove(os.path.join(path, ".incomplete.json"))

    state = ctx.state()
    state["summary"] = {"hr_processed": len(written["hr"]), "lr_adjusted": len(written.get("lr", []))}
    state["directory"] = directory
    state["files"] = written
    state["record"] = ctx.settings_record()
    ctx.on_finish(_write_manifest(ctx.node.id))
    return {}


def _write_manifest(node_id):
    def finish(ctx):
        state = ctx.state(node_id)
        with open(os.path.join(state["directory"], "processed_manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"settings": state["record"], "files": state["files"]}, f, indent=1)
    return finish
