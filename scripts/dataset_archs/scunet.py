# Copied from VSR-Dataset-Creator (dataset_creator/archs/scunet.py), Apache License 2.0,
# used with the author's permission. Unchanged.

"""SCUNet (Swin-Conv UNet, KAIR) with the Prism modifications.

The released Prism H264 SCUNet models (Bubblemint864/Prism-Models) are stock
SCUNet variants except:
  * in_nc=4 — RGB plus a constant "strength" plane (CRF/51),
  * out_nc=3,
  * modified feature/head dimensions — full uses dim=128/head_dim=16 and
    Small uses dim=64/head_dim=8,
  * an extra `m_res` 3x3 conv on the outer skip: m_tail(x + m_res(m_head_out)).
  * symmetric 32px reflect padding around the model, cropped away at output.

The bundled weights were converted from the released ONNX graphs.
"""

from __future__ import annotations

import numpy as np
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn


def _reflect_pad_symmetric(x: torch.Tensor, pad: int = 32) -> torch.Tensor:
    if pad == 0:
        return x

    def indices(length: int, device: torch.device) -> torch.Tensor:
        total = length + 2 * pad
        if length <= 1:
            return torch.zeros(total, dtype=torch.long, device=device)
        period = 2 * length - 2
        idx = torch.arange(-pad, length + pad, dtype=torch.long, device=device)
        mod = torch.remainder(idx, period)
        return torch.where(mod < length, mod, period - mod)

    h, w = x.shape[-2:]
    y_idx = indices(h, x.device)
    x_idx = indices(w, x.device)
    return x.index_select(-2, y_idx).index_select(-1, x_idx)


class WMSA(nn.Module):
    """Self-attention module in Swin Transformer."""

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim**-0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type
        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(
            torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size - 1)
        )
        self.linear = nn.Linear(self.input_dim, self.output_dim)

    def generate_mask(self, h, w, p, shift):
        attn_mask = torch.zeros(
            h, w, p, p, p, p,
            dtype=torch.bool,
            device=self.relative_position_params.device,
        )
        if self.type == "W":
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        return rearrange(attn_mask, "w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)")

    def forward(self, x):
        if self.type != "W":
            x = torch.roll(
                x,
                shifts=(-(self.window_size // 2), -(self.window_size // 2)),
                dims=(1, 2),
            )

        x = rearrange(
            x,
            "b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c",
            p1=self.window_size,
            p2=self.window_size,
        )
        h_windows = x.size(1)
        w_windows = x.size(2)

        x = rearrange(
            x,
            "b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c",
            p1=self.window_size,
            p2=self.window_size,
        )
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(
            qkv, "b nw np (threeh c) -> threeh b nw np c", c=self.head_dim
        ).chunk(3, dim=0)
        sim = torch.einsum("hbwpc,hbwqc->hbwpq", q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), "h p q -> h 1 1 p q")
        if self.type != "W":
            attn_mask = self.generate_mask(
                h_windows, w_windows, self.window_size, shift=self.window_size // 2
            )
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum("hbwij,hbwjc->hbwic", probs, v)
        output = rearrange(output, "h b w p c -> b w p (h c)")
        output = self.linear(output)
        output = rearrange(
            output,
            "b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c",
            w1=h_windows,
            p1=self.window_size,
        )

        if self.type != "W":
            output = torch.roll(
                output,
                shifts=(self.window_size // 2, self.window_size // 2),
                dims=(1, 2),
            )

        return output

    def relative_embedding(self):
        cord = torch.tensor(
            np.array(
                [[i, j] for i in range(self.window_size) for j in range(self.window_size)]
            )
        )
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        return self.relative_position_params[
            :, relation[:, :, 0].long(), relation[:, :, 1].long()
        ]


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, type="W", input_resolution=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ["W", "SW"]
        self.type = type
        if input_resolution <= window_size:
            self.type = "W"

        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.msa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, type="W", input_resolution=None):
        super().__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        assert type in ["W", "SW"]
        self.type = type
        if input_resolution <= window_size:
            self.type = "W"

        self.trans_block = Block(
            trans_dim, trans_dim, head_dim, window_size, self.type, input_resolution
        )
        self.conv1_1 = nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, 1, 1, 0, bias=True)
        self.conv_block = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=False),
        )

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = Rearrange("b c h w -> b h w c")(trans_x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange("b h w c -> b c h w")(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        return x + res


class PrismSCUNet(nn.Module):
    """SCUNet with 4ch input (RGB + strength plane) and an m_res outer skip."""

    def __init__(
        self,
        in_nc: int = 4,
        out_nc: int = 3,
        config: tuple[int, ...] = (4, 4, 4, 4, 4, 4, 4),
        dim: int = 64,
        head_dim: int = 8,
        input_resolution: int = 256,
    ):
        super().__init__()
        self.config = config
        self.dim = dim
        self.head_dim = head_dim
        self.window_size = 8

        def blocks(conv_dim, count, resolution):
            return [
                ConvTransBlock(
                    conv_dim,
                    conv_dim,
                    self.head_dim,
                    self.window_size,
                    "W" if not i % 2 else "SW",
                    resolution,
                )
                for i in range(count)
            ]

        self.m_head = nn.Sequential(nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False))
        self.m_down1 = nn.Sequential(
            *blocks(dim // 2, config[0], input_resolution),
            nn.Conv2d(dim, 2 * dim, 2, 2, 0, bias=False),
        )
        self.m_down2 = nn.Sequential(
            *blocks(dim, config[1], input_resolution // 2),
            nn.Conv2d(2 * dim, 4 * dim, 2, 2, 0, bias=False),
        )
        self.m_down3 = nn.Sequential(
            *blocks(2 * dim, config[2], input_resolution // 4),
            nn.Conv2d(4 * dim, 8 * dim, 2, 2, 0, bias=False),
        )
        self.m_body = nn.Sequential(*blocks(4 * dim, config[3], input_resolution // 8))
        self.m_up3 = nn.Sequential(
            nn.ConvTranspose2d(8 * dim, 4 * dim, 2, 2, 0, bias=False),
            *blocks(2 * dim, config[4], input_resolution // 4),
        )
        self.m_up2 = nn.Sequential(
            nn.ConvTranspose2d(4 * dim, 2 * dim, 2, 2, 0, bias=False),
            *blocks(dim, config[5], input_resolution // 2),
        )
        self.m_up1 = nn.Sequential(
            nn.ConvTranspose2d(2 * dim, dim, 2, 2, 0, bias=False),
            *blocks(dim // 2, config[6], input_resolution),
        )
        self.m_res = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1, bias=False))
        self.m_tail = nn.Sequential(nn.Conv2d(dim, out_nc, 3, 1, 1, bias=False))

    def forward(self, x0):
        h, w = x0.size()[-2:]
        # Matches the released ONNX exactly: symmetric 32 px reflect padding.
        pad = 32
        x0 = _reflect_pad_symmetric(x0, pad)

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + self.m_res(x1))

        return x[:, :, pad : pad + h, pad : pad + w]
