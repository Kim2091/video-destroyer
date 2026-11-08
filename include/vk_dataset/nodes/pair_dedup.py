"""
Dropping sequences whose HR crops look nearly the same, as PepeDP's
deduplication does: each crop is described by a ConvNeXt-S feature vector,
and a crop closer than the threshold (euclidean distance) to one already
kept is a duplicate.

Sequences are judged as they arrive, against every crop kept so far in the
run, most detailed first within a chunk; a dropped sequence takes its whole
LR run with it. The network is the ConvNeXt-S of Meta's ConvNeXt (MIT), as
VSR-Dataset-Creator ships it; its weights are not part of Video Destroyer and are
read from the path the node names. The network is a modified port of
VSR-Dataset-Creator's (Apache-2.0) archs/convnext.py; see NOTICE.md.
"""

import gc
import os

import numpy as np

from ..media import frame_array, to_rgb24
from ..registry import Param, node

DEFAULT_WEIGHTS = "models/convnextS_1kpretrained_official_style.pth"


def _build_convnext():
    import torch
    from torch import nn
    from torch.nn import functional as F

    class LayerNorm(nn.Module):
        def __init__(self, channels, channels_first=False):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))
            self.channels_first = channels_first

        def forward(self, x):
            if not self.channels_first:
                return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-6)
            mean = x.mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt((x - mean).pow(2).mean(1, keepdim=True) + 1e-6)
            return self.weight[:, None, None] * x + self.bias[:, None, None]

    class Block(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.dwconv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
            self.norm = LayerNorm(channels)
            self.pwconv1 = nn.Linear(channels, 4 * channels)
            self.act = nn.GELU()
            self.pwconv2 = nn.Linear(4 * channels, channels)
            self.gamma = nn.Parameter(1e-6 * torch.ones(channels))

        def forward(self, x):
            y = self.dwconv(x).permute(0, 2, 3, 1)
            y = self.pwconv2(self.act(self.pwconv1(self.norm(y))))
            return x + (self.gamma * y).permute(0, 3, 1, 2)

    class ConvNeXtSmall(nn.Module):
        def __init__(self):
            super().__init__()
            depths, dims = (3, 3, 27, 3), (96, 192, 384, 768)
            self.downsample_layers = nn.ModuleList(
                [nn.Sequential(nn.Conv2d(3, dims[0], kernel_size=4, stride=4), LayerNorm(dims[0], True))]
                + [nn.Sequential(LayerNorm(dims[i], True), nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2))
                   for i in range(3)])
            self.stages = nn.ModuleList(nn.Sequential(*(Block(dims[i]) for _ in range(depths[i]))) for i in range(4))

        def forward(self, x):
            for downsample, stage in zip(self.downsample_layers, self.stages):
                x = stage(downsample(x))
            return x.mean((-2, -1))

    return ConvNeXtSmall()


class Embedder:
    """ConvNeXt-S features of an image shrunk to a quarter of its size, as PepeDP's dedup takes them."""

    def __init__(self, weights, device):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.model = _build_convnext()
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model.eval().to(self.device)

    def __call__(self, rgb):
        torch = self.torch
        from torch.nn import functional as F
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None].to(self.device)
            h, w = rgb.shape[:2]
            x = F.interpolate(x, size=(max(1, h // 4), max(1, w // 4)), mode="bilinear", antialias=True,
                              align_corners=False)
            with torch.autocast(self.device.type, torch.float16, enabled=self.device.type == "cuda"):
                return self.model(x).float().flatten()


def resolve_weights(path):
    """A relative path is taken from the folder that holds the vk_dataset package."""
    if not os.path.isabs(path):
        include = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(include, path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"the ConvNeXt-S weights were not found at {path}; this is "
                                "convnextS_1kpretrained_official_style.pth from VSR-Dataset-Creator's models folder")
    return path


@node("dedup_sequences", category="Sequences", label="Drop Near-Duplicate Sequences",
      description="Drops sequences whose HR crop is perceptually almost the same as one already kept "
                  "(ConvNeXt-S features, PepeDP's method).",
      inputs={"sequences": "sequences"}, outputs={"sequences": "sequences"},
      params=[
          Param("weights", "path", DEFAULT_WEIGHTS,
                help="convnextS_1kpretrained_official_style.pth; a relative path is read from the folder "
                     "that holds the dataset engine."),
          Param("threshold", "float", 1.5, help="Feature distance below which two crops are duplicates.", min=0.0),
          Param("clip", "str", "hr", help="The clip whose crops are compared."),
          Param("backend", "choice", "auto", choices=["auto", "cuda", "cpu"],
                help="'auto' uses CUDA when PyTorch sees a GPU, else the CPU."),
      ])
def dedup_sequences(ctx, sequences, *, weights, threshold, clip, backend):
    from .pair_align import resolve_backend
    import torch
    state = ctx.state()
    if "embedder" not in state:
        state["embedder"] = Embedder(resolve_weights(weights), resolve_backend(backend))
        state["kept"] = {}
        ctx.on_finish(_release(ctx.node.id))
    embedder = state["embedder"]
    # Variants of a writer share their HR, so each variant is compared only with its own.
    kept = state["kept"].setdefault(ctx.variant, [])
    source = to_rgb24(sequences.clips[clip])

    def detail(item):
        return item.info.get("complexity", 0.0)

    kept_items, dropped = [], 0
    for item in sorted(sequences.items, key=detail, reverse=True):
        frames = item.frames[clip]
        image = frame_array(source.get_frame(frames[len(frames) // 2]))
        crop = item.crops.get(clip)
        if crop:
            left, top, w, h = crop
            image = image[top:top + h, left:left + w]
        features = embedder(image.astype(np.float32) / 255.0)
        if kept:
            distance = float(torch.cdist(features[None], torch.stack(kept))[0].min())
            if distance < threshold:
                dropped += 1
                continue
            item.info["dedup_distance"] = round(distance, 4)
        kept.append(features)
        kept_items.append(item)
    if dropped:
        summary = state.setdefault("summary", {})
        summary["duplicates_dropped"] = summary.get("duplicates_dropped", 0) + dropped
    order = {id(item): i for i, item in enumerate(sequences.items)}
    return {"sequences": sequences.replace(sorted(kept_items, key=lambda item: order[id(item)]))}


def _release(node_id):
    def finish(ctx):
        state = ctx.state(node_id)
        state.pop("embedder", None)
        state.pop("kept", None)
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return finish
