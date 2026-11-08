"""
The analogue video chain: ntsc-simulator's NTSC composite and VHS path as
four device nodes, run through its VapourSynth plugin (native/vs-ntsc),
which the app ships in vk_dataset/plugins.

    Composite Encode -> VHS Tape -> Playback Deck -> Composite Decode

Encode turns a clip into a signal; Tape and Deck take a signal and give one
back; Decode turns a signal into a clip. A signal is only a description of
the chain so far: the source clip, the tape grade and deck, the settings
each device set, the dubs and the seed. Nothing is simulated until Decode
runs the whole chain once through the plugin, so the result is exactly what
the simulator's CLI gives for the same settings. A signal is never changed
in place, so one Tape may feed two Decks and each gets its own chain.

The simulator runs one tape and one deck per chain. A Tape fed by a Tape
records a dub of it: the plugin adds tape generations, each with its own
noise and dropouts on the same kind of tape, so a dub takes only its
Generations setting. Every simulator parameter belongs to one device (the
*_PARAMS sets below); a device sets only its own, and the tape grade and the
deck profile fill in the rest as on the CLI (explicit values, then the deck,
then the tape).

Each chunk draws its own settings: the tape grade, the ranges of the shown
settings (recorded in the manifest) and the seed, from which the simulator
draws the tape's own ranges and every frame's noise. Frames do not depend on
the order they are read in. The tape holds about 480 lines whatever the
frame size, so the chain keeps the clip's size and looks like an SD tape
scaled to it; it is meant for LR clips near SD size.
"""

import dataclasses
import functools
import os
import sys

import vapoursynth as vs

from ..framefx import from_rgbs, to_rgbs
from ..registry import Param, node

core = vs.core

PLUGIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plugins",
                      "vsntsc.dll" if sys.platform == "win32" else "libvsntsc.so")

TAPES = ["lq", "mq", "hq"]
# Deck node choice -> the simulator's deck profile; the consumer deck is the
# one the tape grades already describe.
DECKS = {"consumer": "", "svhs-tbc": "deck-svhs-tbc", "edit": "deck-edit", "vhs-decode": "deck-vhs-decode"}
COMBS = ["auto", "trap", "2sample", "1h", "2d", "3d"]
OUTPUTS = {"auto": None, "composite": "composite", "s-video": "svideo"}

# The simulator parameters each device owns, by CLI flag.
ENCODE_PARAMS = {
    # The source chain's edge enhancement and caption data, then off-air reception.
    "edge-ringing", "ringing-freq", "ringing-direction", "vbi-data",
    "noise", "ghost", "ghost-delay", "ghost-phase", "ghost-rolloff-mhz", "ghost-dynamic", "ghost-dynamic-rate",
    "ghost-multi", "attenuation", "reception-before-tape",
}
RECEPTION = {"noise", "ghost", "ghost-delay", "ghost-phase", "ghost-rolloff-mhz", "ghost-dynamic",
             "ghost-dynamic-rate", "ghost-multi", "attenuation"}
TAPE_PARAMS = {
    "tape-model", "vhs-luma-bw", "color-under-bw", "white-clip", "dark-clip", "nl-emphasis", "tape-snr", "rf-loss",
    "tracking-bars", "luma-crosstalk", "ep-interleave", "bw-inversion", "head-level-step", "generations",
    "head-smear", "tape-trail",
    "chroma-noise", "chroma-noise-bw", "chroma-crosstalk", "chroma-phase-noise", "chroma-delay", "chroma-delay-v",
    "luma-noise", "tape-dropout-rate", "dropouts", "dropout-span", "tape-dropout-len",
    "jitter", "wobble", "wobble-rate", "v-jitter", "head-switch", "head-switch-lines", "head-switch-point",
    "head-switch-after", "flagging", "top-band", "temporal-coherence",
}
DECK_PARAMS = {
    "line-tbc", "frame-tbc", "deck-dnr", "deck-nc", "deck-ynr", "deck-cnr", "deck-sharpness", "chroma-vblur", "doc",
    "video-out",
}
DECODE_PARAMS = {"comb", "capture-peaking", "capture-peaking-mhz", "legacy-burst-phase"}
DEVICE_PARAMS = {"Composite Encode": ENCODE_PARAMS, "VHS Tape": TAPE_PARAMS, "Playback Deck": DECK_PARAMS,
                 "Composite Decode": DECODE_PARAMS}

AUTO = "Empty keeps the tape's own value"
TAPE_SETTINGS = {
    # node setting: (simulator parameter, help, min, max)
    "tape_snr": ("tape-snr", "Tape signal-to-noise ratio in dB; lower is noisier (worn EP about 30, SP 37).", 25.0, 55.0),
    "luma_bandwidth": ("vhs-luma-bw", "Luma bandwidth in MHz; lower is softer.", 0.5, 4.2),
    "colour_bandwidth": ("color-under-bw", "Colour bandwidth in kHz; lower smears colour more.", 100.0, 1000.0),
    "chroma_noise": ("chroma-noise", "Colour noise: blotches in flat areas (0.01 to 0.03).", 0.0, 0.1),
    "chroma_delay": ("chroma-delay", "Colour shift to the right in pixels (1 to 3).", -4.0, 8.0),
    "dropouts": ("dropouts", "Dropouts per minute: white or black streaks (5 to 100).", 0.0, 200.0),
    "wobble": ("wobble", "Sideways wobble of the picture in pixels.", 0.0, 6.0),
    "head_switch": ("head-switch", "Head-switching tear at the bottom of the frame (0 to 1).", 0.0, 1.0),
    "tracking": ("tracking-bars", "A rolling band of noise from tracking that is slightly off (0 to 1).", 0.0, 1.0),
}
DECK_SETTINGS = {
    "sharpness": ("deck-sharpness", "Picture control: positive sharpens with thin halos, negative softens.", -1.0, 1.0),
    "noise_canceller": ("deck-nc", "Removes noise and fine detail below this many IRE (0 is off).", 0.0, 15.0),
    "noise_reduction": ("deck-dnr", "Averages still areas over frames (0 is off).", 0.0, 1.0),
    "tbc": ("line-tbc", "Line time-base corrector: straightens wobble and jitter (0 is off).", 0.0, 1.0),
}
DECK_AUTO = "Empty keeps the deck's own value"
LABELS = {"tape_snr": "Tape S/N", "tbc": "TBC"}


def ntsc():
    """The plugin's namespace, loading the bundled plugin if it is not loaded yet."""
    if not hasattr(core, "ntsc"):
        if not os.path.exists(PLUGIN):
            raise RuntimeError(f"the analogue video plugin is missing ({PLUGIN}); build it with npm run build:native")
        core.std.LoadPlugin(PLUGIN)
    return core.ntsc


def arg(flag):
    """A simulator CLI flag as the plugin's argument name."""
    return flag.replace("-", "_")


def number(value):
    """A drawn or fixed number in the plugin's text form."""
    return repr(float(value))


def parse_more(text, device):
    """'name=value name=value' as plugin arguments, for parameters `device` owns."""
    out = {}
    own = DEVICE_PARAMS[device]
    for item in text.replace(";", " ").split():
        name, sep, value = item.partition("=")
        name, value = name.strip().replace("_", "-"), value.strip()
        if not sep or not name or not value:
            raise ValueError(f"more settings must read name=value, got {item!r}")
        if name not in own:
            owner = next((d for d, params in DEVICE_PARAMS.items() if name in params), None)
            raise ValueError(f"{name} is a setting of {owner}, not of {device}" if owner
                             else f"{name} is not a setting of {device}")
        out[arg(name)] = value
    return out


@dataclasses.dataclass(frozen=True)
class Signal:
    """The chain so far, from the Encode node to the last device."""
    source: vs.VideoNode
    like: vs.VideoNode
    seed: int
    args: tuple = ()
    tape: str = ""
    deck: str | None = None
    dubs: int = 0

    def add(self, args=None, **changes):
        merged = {**dict(self.args), **(args or {})}
        return dataclasses.replace(self, args=tuple(sorted(merged.items())), **changes)

    def simulate(self, **decode):
        """The chain's picture, decoded with `decode` (plugin arguments), in the source clip's format."""
        out = ntsc().Simulate(self.source, preset=self.tape, deck=DECKS.get(self.deck or "consumer", ""),
                              seed=self.seed, dubs=self.dubs, **dict(self.args), **decode)
        return out if self.source is self.like else from_rgbs(out, self.like)

    @functools.cached_property
    def preview(self):
        """The chain so far decoded with plain settings: the tape's consumer deck if no deck yet, auto comb."""
        return self.simulate()


def own_values(settings, values):
    """Plugin arguments for the shown settings that are set, by simulator parameter."""
    return {arg(settings[name][0]): number(value) for name, value in values.items() if value is not None}


@node("ntsc_encode", category="Degrade", label="Composite Encode",
      inputs={"clip": "clip"}, outputs={"signal": "signal"},
      params=[
          Param("sharpening", "float", None, random=True, min=0.0, max=3.0,
                help="Edge enhancement from the source chain: halos beside sharp edges (0.2 to 0.4). " + AUTO + "."),
          Param("snow", "float", 0.0, random=True, min=0.0, max=0.3,
                help="Off-air reception noise (0.05 subtle, 0.2 heavy); the tape records it. 0 is off."),
          Param("ghost", "float", 0.0, random=True, min=0.0, max=1.0,
                help="A faint delayed copy of the picture to the right (0 to 1). 0 is off."),
          Param("ghost_delay", "float", 2.0, random=True, min=0.0, max=20.0,
                help="How late the ghost arrives, in microseconds; larger is further right."),
          Param("weak_signal", "float", 0.0, random=True, min=0.0, max=1.0,
                help="A weak signal: less contrast and colour (0 to 1). 0 is off."),
          Param("more", "str", "", label="More settings",
                help="Other source and reception settings as name=value with spaces between, e.g. "
                     "'vbi-data=0.5 ghost-phase=180'. Names are the simulator's CLI flags."),
      ])
def ntsc_encode(ctx, clip, *, sharpening, snow, ghost, ghost_delay, weak_signal, more):
    args = parse_more(more, "Composite Encode")
    if sharpening is not None:
        args["edge_ringing"] = number(sharpening)
    for flag, value in (("noise", snow), ("ghost", ghost), ("attenuation", weak_signal)):
        if value:
            args[arg(flag)] = number(value)
    if ghost:
        args["ghost_delay"] = number(ghost_delay)
    # Reception comes before the tape: the recorder tapes what it received.
    if any(arg(flag) in args for flag in RECEPTION):
        args.setdefault("reception_before_tape", "true")
    native = clip.format.id in (vs.RGB24, vs.RGB48, vs.RGBS)
    seed = ctx.rng.getrandbits(63)
    ctx.note("seed", seed)
    return {"signal": Signal(clip if native else to_rgbs(clip), clip, seed).add(args)}


@node("ntsc_tape", category="Degrade", label="VHS Tape", probability=True,
      inputs={"signal": "signal"}, outputs={"signal": "signal"},
      params=[
          Param("tape", "choice", "any", choices=["any", *TAPES],
                help="lq: worn EP tape, mq: rental tape, hq: clean SP recording. 'any' picks one per chunk."),
          Param("generations", "int", None, random=True, min=1, max=4,
                help="1 is the original recording, 2 a dub of it, and so on: noise adds up. On a Tape fed "
                     "by a Tape, the generations this dub adds (empty is 1). " + AUTO + "."),
          *[Param(name, "float", None, random=True, min=low, max=high, label=LABELS.get(name, ""),
                  help=f"{text} {AUTO}.")
            for name, (_, text, low, high) in TAPE_SETTINGS.items()],
          Param("more", "str", "", label="More settings",
                help="Other tape settings as name=value with spaces between, e.g. 'rf-loss=0.5 v-jitter=1'. "
                     "Names are the simulator's CLI flags."),
      ])
def ntsc_tape(ctx, signal, *, tape, generations, more, **settings):
    if signal.deck is not None:
        raise ValueError("a VHS Tape comes before the Playback Deck: a chain plays through one deck, after "
                         "its last tape")
    if signal.tape:
        # A dub copies the tape before it, on the same kind of tape.
        if tape != "any" or more.strip() or any(v is not None for v in settings.values()):
            raise ValueError("this VHS Tape records a dub of the tape before it, which keeps that tape's "
                             "settings: set only its Generations")
        dubs = 1 if generations is None else max(1, int(round(generations)))
        seed = ctx.rng.getrandbits(63)
        ctx.note("dub", dubs)
        ctx.note("seed", seed)
        return {"signal": signal.add(dubs=signal.dubs + dubs, seed=seed)}
    if tape not in ["any", *TAPES]:
        raise ValueError(f"tape must be any, {', '.join(TAPES)}, got {tape!r}")
    tape = ctx.rng.choice(TAPES) if tape == "any" else tape
    args = parse_more(more, "VHS Tape")
    args.update(own_values(TAPE_SETTINGS, settings))
    if generations is not None:
        args["generations"] = str(max(1, int(round(generations))))
    seed = ctx.rng.getrandbits(63)
    ctx.note("tape", tape)
    ctx.note("seed", seed)
    return {"signal": signal.add(args, tape=tape, seed=seed)}


@node("ntsc_deck", category="Degrade", label="Playback Deck", probability=True,
      inputs={"signal": "signal"}, outputs={"signal": "signal"},
      params=[
          Param("deck", "choice", "consumer", choices=list(DECKS),
                help="consumer: an ordinary VCR (noise canceller, mild sharpening). svhs-tbc: high-end S-VHS "
                     "deck with TBC and noise reduction. edit: sharpening and noise reducers off. vhs-decode: "
                     "an RF capture, no deck electronics."),
          *[Param(name, "float", None, random=True, min=low, max=high, label=LABELS.get(name, ""),
                  help=f"{text} {DECK_AUTO}.")
            for name, (_, text, low, high) in DECK_SETTINGS.items()],
          Param("output", "choice", "auto", choices=list(OUTPUTS),
                help="The cable to the TV or capture card. auto: the deck's own (S-Video on svhs-tbc and "
                     "vhs-decode). Composite adds rainbows and dot crawl."),
          Param("more", "str", "", label="More settings",
                help="Other deck settings as name=value with spaces between, e.g. 'frame-tbc=1 doc=composite'. "
                     "Names are the simulator's CLI flags."),
      ])
def ntsc_deck(ctx, signal, *, deck, output, more, **settings):
    if signal.deck is not None:
        raise ValueError("a chain plays through one Playback Deck; this one follows another")
    if deck not in DECKS:
        raise ValueError(f"deck must be one of {', '.join(DECKS)}, got {deck!r}")
    if output not in OUTPUTS:
        raise ValueError(f"output must be one of {', '.join(OUTPUTS)}, got {output!r}")
    args = parse_more(more, "Playback Deck")
    args.update(own_values(DECK_SETTINGS, settings))
    if OUTPUTS[output]:
        args["video_out"] = OUTPUTS[output]
    ctx.note("deck", deck)
    return {"signal": signal.add(args, deck=deck)}


@node("ntsc_decode", category="Degrade", label="Composite Decode",
      inputs={"signal": "signal"}, outputs={"clip": "clip"},
      params=[
          Param("comb", "choice", "auto", choices=COMBS,
                help="How the TV or capture card splits luma and colour on composite. auto: the chain's own "
                     "(a notch trap, a 3D comb with svhs-tbc). 2d and 3d leave less dot crawl."),
          Param("peaking", "float", 0.0, random=True, min=0.0, max=2.0,
                help="The capture chip's luma peaking: crisper grain and thin halos (0.25 is a mild boost). "
                     "0 is off."),
          Param("peaking_mhz", "float", 2.6, random=True, min=1.5, max=4.5, label="Peaking centre",
                help="Centre frequency of the peaking in MHz."),
      ])
def ntsc_decode(ctx, signal, *, comb, peaking, peaking_mhz):
    if comb not in COMBS:
        raise ValueError(f"comb must be one of {', '.join(COMBS)}, got {comb!r}")
    args = {} if comb == "auto" else {"comb": comb}
    if peaking:
        args.update(capture_peaking=number(peaking), capture_peaking_mhz=number(peaking_mhz))
    return {"clip": signal.simulate(**args)}
