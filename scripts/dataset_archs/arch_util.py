# Copied from VSR-Dataset-Creator (dataset_creator/archs/arch_util.py), Apache License 2.0,
# used with the author's permission. Unchanged.

"""Minimal stand-ins for the traiNNer-redux arch utilities used by GateRV3.

Only the upsampler variants that can actually appear in inference state dicts
are implemented. The bundled denoise model is scale=1 and does not use
UniUpsample at all, but generic GateRV3 checkpoints may.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

SampleMods = Literal[
    "conv",
    "pixelshuffledirect",
    "pixelshuffle",
    "nearest+conv",
    "dysample",
]

_SAMPLE_MODE_ORDER = (
    "conv",
    "pixelshuffledirect",
    "pixelshuffle",
    "nearest+conv",
    "dysample",
)


class UniUpsample(nn.Sequential):
    def __init__(
        self,
        upsample: SampleMods,
        scale: int = 2,
        in_dim: int = 64,
        out_dim: int = 3,
        mid_dim: int = 64,
        group: int = 4,
    ) -> None:
        m: list[nn.Module] = []
        if scale == 1 or upsample == "conv":
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        elif upsample == "pixelshuffledirect":
            m.extend(
                [nn.Conv2d(in_dim, out_dim * scale**2, 3, 1, 1), nn.PixelShuffle(scale)]
            )
        elif upsample == "pixelshuffle":
            m.extend([nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)])
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        [nn.Conv2d(mid_dim, 4 * mid_dim, 3, 1, 1), nn.PixelShuffle(2)]
                    )
            elif scale == 3:
                m.extend([nn.Conv2d(mid_dim, 9 * mid_dim, 3, 1, 1), nn.PixelShuffle(3)])
            else:
                raise ValueError(
                    f"scale {scale} is not supported by pixelshuffle. Use 2^n or 3."
                )
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        elif upsample == "nearest+conv":
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        [
                            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                            nn.Upsample(scale_factor=2),
                            nn.LeakyReLU(0.2, inplace=True),
                        ]
                    )
                m.extend(
                    [nn.Conv2d(in_dim, in_dim, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)]
                )
            elif scale == 3:
                m.extend(
                    [
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.Upsample(scale_factor=scale),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                    ]
                )
            else:
                raise ValueError(
                    f"scale {scale} is not supported by nearest+conv. Use 2^n or 3."
                )
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        else:
            raise NotImplementedError(
                f"Upsample mode '{upsample}' is not supported in this build."
            )
        super().__init__(*m)
        self.register_buffer(
            "MetaUpsample",
            torch.tensor(
                [
                    1,
                    _SAMPLE_MODE_ORDER.index(upsample),
                    scale,
                    in_dim,
                    out_dim,
                    mid_dim,
                    group,
                ],
                dtype=torch.uint8,
            ),
        )
