# Copied from VSR-Dataset-Creator (dataset_creator/archs/bgcc_v4.py), Apache License 2.0,
# used with the author's permission. Unchanged.

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


class ConvAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=True,
        )
        self.act = nn.PReLU(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.conv(x))


class DilatedResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = ConvAct(channels, channels, dilation=dilation)
        self.conv2 = nn.Conv2d(
            channels, channels, 3, padding=dilation, dilation=dilation, bias=True
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv2(self.conv1(x))


class ContextStage(nn.Module):
    def __init__(self, channels: int, n_blocks: int) -> None:
        super().__init__()
        dilations = (1, 2, 4, 2)
        self.body = nn.Sequential(
            *[
                DilatedResBlock(channels, dilation=dilations[i % len(dilations)])
                for i in range(n_blocks)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class StructureStats(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    @staticmethod
    def luma(x: Tensor) -> Tensor:
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    def edge_map(self, x: Tensor) -> Tensor:
        luma = self.luma(x)
        grad_x = F.conv2d(luma, self.sobel_x, padding=1)
        grad_y = F.conv2d(luma, self.sobel_y, padding=1)
        edge = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
        return edge / (edge.amax(dim=(2, 3), keepdim=True) + 1e-6)


class GlobalColorAlign(nn.Module):
    """Whole-pair tone/grade alignment estimated from reliability-weighted stats.

    Predicts a global 3x3 color matrix + bias plus per-channel hinge curves and
    applies them to HR. Because the parameters come from reliability-weighted
    pooling over the entire pair, local degradations (halos, bleed) average out
    and exposure/contrast mismatch is absorbed here instead of leaking into the
    spatially-varying stages.
    """

    def __init__(self, channels: int, n_knots: int = 8) -> None:
        super().__init__()
        self.n_knots = n_knots
        knots = torch.linspace(1.0 / (n_knots + 1), n_knots / (n_knots + 1), n_knots)
        self.register_buffer("knots", knots, persistent=False)
        eye = torch.eye(3).reshape(1, 9)
        self.register_buffer("eye", eye, persistent=False)

        n_params = 9 + 3 + 3 * n_knots
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, n_params),
        )
        final = self.mlp[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        if final.bias is not None:
            nn.init.zeros_(final.bias)

    def forward(self, pooled: Tensor, hr: Tensor) -> Tensor:
        b, _, h, w = hr.shape
        params = self.mlp(pooled)
        mat = (params[:, :9] + self.eye).view(b, 3, 3)
        bias = params[:, 9:12].view(b, 3, 1, 1)
        slopes = params[:, 12:].view(b, 3, self.n_knots) / self.n_knots

        y = torch.bmm(mat, hr.flatten(2)).view(b, 3, h, w) + bias
        # Per-channel piecewise-linear residual curve (hinge basis, identity at init).
        curve = torch.zeros_like(y)
        for k in range(self.n_knots):
            curve = curve + slopes[:, :, k].view(b, 3, 1, 1) * F.relu(
                y - self.knots[k]
            )
        return y + curve


class PaletteRouter(nn.Module):
    """Reliability-weighted palette slots with HR-only queries.

    Slots soft-cluster the evidence features by appearance (not by spatial
    pooling), with low-reliability LR locations contributing less to each
    slot's gathered correction. Pixels then query the slots using HR-only
    appearance features plus a coarse position embedding, so a pixel whose LR
    counterpart is corrupted or occluded still retrieves the correction
    observed wherever its color appears with clean evidence.
    """

    def __init__(
        self,
        kv_channels: int,
        q_channels: int,
        out_channels: int,
        num_slots: int = 32,
        slot_dim: int = 64,
    ) -> None:
        super().__init__()
        self.slot_dim = slot_dim
        self.slots = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, slot_dim, 16, 16) * 0.02)
        self.to_k = nn.Conv2d(kv_channels, slot_dim, 1, bias=True)
        self.to_v = nn.Conv2d(kv_channels, slot_dim, 1, bias=True)
        self.to_q = nn.Conv2d(q_channels, slot_dim, 1, bias=True)
        self.to_out = nn.Conv2d(slot_dim, out_channels, 1, bias=True)

    def forward(self, kv_feat: Tensor, q_feat: Tensor, reliability: Tensor) -> Tensor:
        b, _, h_kv, w_kv = kv_feat.shape
        _, _, h_q, w_q = q_feat.shape
        scale = 1.0 / math.sqrt(self.slot_dim)

        pos_kv = F.interpolate(
            self.pos, size=(h_kv, w_kv), mode="bilinear", align_corners=False
        )
        pos_q = F.interpolate(
            self.pos, size=(h_q, w_q), mode="bilinear", align_corners=False
        )

        k = (self.to_k(kv_feat) + pos_kv).flatten(2)  # (B, D, N_kv)
        v = self.to_v(kv_feat).flatten(2).transpose(1, 2)  # (B, N_kv, D)
        log_rel = torch.log(reliability.flatten(2) + 1e-4)  # (B, 1, N_kv)

        slots = self.slots.expand(b, -1, -1)  # (B, S, D)
        gather_attn = torch.softmax(
            torch.bmm(slots, k) * scale + log_rel, dim=-1
        )  # (B, S, N_kv)
        slot_vals = torch.bmm(gather_attn, v)  # (B, S, D)

        q = (self.to_q(q_feat) + pos_q).flatten(2).transpose(1, 2)  # (B, N_q, D)
        route_attn = torch.softmax(
            torch.bmm(q, slots.transpose(1, 2)) * scale, dim=-1
        )  # (B, N_q, S)
        routed = torch.bmm(route_attn, slot_vals)  # (B, N_q, D)
        routed = routed.transpose(1, 2).reshape(b, self.slot_dim, h_q, w_q)
        return self.to_out(routed)


class GuidedCoeffUpsampler(nn.Module):
    """Joint-bilateral upsampling of coarse transform coefficients.

    Each HR pixel blends the 3x3 neighborhood of coarse coefficient cells,
    weighted by HR color similarity, so coefficients snap to the correct side
    of color boundaries without ever varying per HR pixel on their own.
    """

    def __init__(self, sharpness: float = 20.0) -> None:
        super().__init__()
        self.sharpness = sharpness

    @staticmethod
    def _shift(x: Tensor, dy: int, dx: int) -> Tensor:
        _, _, h, w = x.shape
        x_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
        y0 = 1 + dy
        x0 = 1 + dx
        return x_pad[:, :, y0 : y0 + h, x0 : x0 + w]

    def forward(self, coeffs: Tensor, guide_hr: Tensor, guide_lr: Tensor) -> Tensor:
        size = guide_hr.shape[-2:]
        num = torch.zeros(
            coeffs.shape[0],
            coeffs.shape[1],
            *size,
            dtype=coeffs.dtype,
            device=coeffs.device,
        )
        den = torch.zeros(
            coeffs.shape[0], 1, *size, dtype=coeffs.dtype, device=coeffs.device
        )
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                c_up = F.interpolate(
                    self._shift(coeffs, dy, dx),
                    size=size,
                    mode="bilinear",
                    align_corners=False,
                )
                g_up = F.interpolate(
                    self._shift(guide_lr, dy, dx),
                    size=size,
                    mode="bilinear",
                    align_corners=False,
                )
                weight = torch.exp(
                    -self.sharpness
                    * (guide_hr - g_up).abs().mean(dim=1, keepdim=True)
                )
                num = num + c_up * weight
                den = den + weight
        return num / (den + 1e-6)


class BGCCv4(nn.Module):
    """Palette-routed color transform for degraded anime HR/LR pairs.

    Four mechanisms target the observed v1-v3 failure modes:

    1. Global tone/grade alignment (GlobalColorAlign) absorbs exposure and
       contrast mismatch before any spatial stage sees the pair, so per-pixel
       paths no longer encode global shifts as uneven local residuals (v3's
       contrast amplification).
    2. A learned reliability map down-weights degraded LR evidence (halos,
       bleed, dot crawl) in both the global pooling and the palette gather,
       replacing v2/v3's hand-crafted unreliability cue that boosted palette
       injection near edges (v3's halo-as-brightness regression).
    3. PaletteRouter soft-clusters evidence by appearance with HR-only queries,
       letting occluded or crevice pixels retrieve corrections from anywhere
       the same color appears with clean evidence (the obscured-color issue).
    4. Output is a coarse, smooth field of 3x4 affine coefficients applied to
       the aligned HR via joint-bilateral upsampling — a transform of HR colors
       rather than a free residual, so flat HR regions stay flat regardless of
       LR bleed gradients (v1's gradientized transfer).
    """

    def __init__(
        self,
        feat: int = 32,
        n_blocks: int = 3,
        num_slots: int = 32,
        slot_dim: int = 64,
        n_knots: int = 8,
        coeff_scale: float = 0.5,
        refine_bound: float = 0.1,
        jbu_sharpness: float = 20.0,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.coeff_scale = coeff_scale
        self.refine_bound = refine_bound
        self.use_checkpoint = use_checkpoint

        self.stats = StructureStats()
        # hr, lr_up, lr_up - hr, hr_luma, hr_edge, lr_edge, edge_diff
        self.in_channels = 3 + 3 + 3 + 1 + 1 + 1 + 1

        # Evidence trunk (sees both HR and LR): full res -> 1/2 -> 1/4.
        self.enc0 = nn.Sequential(
            ConvAct(self.in_channels, feat), ContextStage(feat, n_blocks)
        )
        self.down1 = nn.Conv2d(feat, feat * 2, 3, stride=2, padding=1, bias=True)
        self.enc1 = nn.Sequential(nn.PReLU(feat * 2), ContextStage(feat * 2, n_blocks))
        self.down2 = nn.Conv2d(feat * 2, feat * 4, 3, stride=2, padding=1, bias=True)
        self.enc2 = nn.Sequential(
            nn.PReLU(feat * 4), ContextStage(feat * 4, n_blocks + 1)
        )

        self.reliability_head = nn.Sequential(
            ConvAct(feat * 4, feat),
            nn.Conv2d(feat, 1, 1, bias=True),
        )

        self.global_align = GlobalColorAlign(feat * 4, n_knots=n_knots)

        # HR-only appearance encoder for router queries: full res -> 1/2.
        self.app_enc0 = nn.Sequential(ConvAct(5, feat), ContextStage(feat, 1))
        self.app_down = nn.Conv2d(feat, feat * 2, 3, stride=2, padding=1, bias=True)
        self.app_enc1 = nn.Sequential(
            nn.PReLU(feat * 2), ContextStage(feat * 2, n_blocks)
        )

        self.router = PaletteRouter(
            kv_channels=feat * 4 + 3,
            q_channels=feat * 2,
            out_channels=feat * 2,
            num_slots=num_slots,
            slot_dim=slot_dim,
        )

        # Coefficient decoder: 1/4 context fused with 1/2 routed + appearance.
        self.dec4 = nn.Sequential(
            ConvAct(feat * 4 + 3, feat * 4), ContextStage(feat * 4, n_blocks)
        )
        self.dec4_up = nn.Conv2d(feat * 4, feat * 2, 3, padding=1, bias=True)
        self.fuse2 = nn.Sequential(
            ConvAct(feat * 6, feat * 2), ContextStage(feat * 2, n_blocks)
        )
        self.coeff_head = nn.Sequential(
            ConvAct(feat * 2, feat * 2),
            nn.Conv2d(feat * 2, 12, 3, padding=1, bias=True),
        )
        self.upsampler = GuidedCoeffUpsampler(sharpness=jbu_sharpness)

        # Full-res refinement, strictly bounded.
        self.refine = nn.Sequential(
            ConvAct(7, feat),
            ContextStage(feat, max(n_blocks - 1, 1)),
            nn.Conv2d(feat, 3, 3, padding=1, bias=True),
        )

        final_coeff = self.coeff_head[-1]
        assert isinstance(final_coeff, nn.Conv2d)
        nn.init.zeros_(final_coeff.weight)
        if final_coeff.bias is not None:
            nn.init.zeros_(final_coeff.bias)

        final_refine = self.refine[-1]
        assert isinstance(final_refine, nn.Conv2d)
        nn.init.zeros_(final_refine.weight)
        if final_refine.bias is not None:
            nn.init.zeros_(final_refine.bias)

    def _stage(self, module: nn.Module, x: Tensor) -> Tensor:
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            out = checkpoint(module, x, use_reentrant=False)
            assert isinstance(out, Tensor)
            return out
        return module(x)

    def forward(self, hr: Tensor, lr: Tensor) -> Tensor:
        b, _, h_hr, w_hr = hr.shape
        lr_up = F.interpolate(
            lr, size=(h_hr, w_hr), mode="bicubic", align_corners=False
        )

        hr_luma = self.stats.luma(hr)
        hr_edge = self.stats.edge_map(hr)
        lr_edge = self.stats.edge_map(lr_up)
        edge_diff = (lr_edge - hr_edge).abs()

        x = torch.cat(
            [hr, lr_up, lr_up - hr, hr_luma, hr_edge, lr_edge, edge_diff], dim=1
        )
        x0 = self._stage(self.enc0, x)
        x1 = self._stage(self.enc1, self.down1(x0))
        evid = self._stage(self.enc2, self.down2(x1))

        reliability = torch.sigmoid(self.reliability_head(evid))  # (B, 1, h/4, w/4)

        # Global alignment from reliability-weighted pair statistics.
        rel_sum = reliability.sum(dim=(2, 3))
        pooled = (evid * reliability).sum(dim=(2, 3)) / (rel_sum + 1e-4)
        hr_base = self.global_align(pooled, hr)

        # Observed correction evidence after global alignment.
        obs = F.interpolate(
            lr_up - hr_base, size=evid.shape[-2:], mode="bilinear", align_corners=False
        )
        kv_feat = torch.cat([evid, obs], dim=1)

        # HR-only appearance for router queries (no LR pollution).
        app = self._stage(self.app_enc0, torch.cat([hr_base, hr_luma, hr_edge], dim=1))
        app = self._stage(self.app_enc1, self.app_down(app))  # (B, 2f, h/2, w/2)

        routed = self.router(kv_feat, app, reliability)  # (B, 2f, h/2, w/2)

        d4 = self._stage(self.dec4, kv_feat)
        d2 = F.interpolate(
            d4, size=app.shape[-2:], mode="bilinear", align_corners=False
        )
        d2 = self._stage(self.fuse2, torch.cat([self.dec4_up(d2), routed, app], dim=1))

        coeffs = torch.tanh(self.coeff_head(d2)) * self.coeff_scale  # (B, 12, h/2, w/2)
        guide_lr = F.interpolate(
            hr_base, size=coeffs.shape[-2:], mode="bilinear", align_corners=False
        )
        m = self.upsampler(coeffs, hr_base, guide_lr)  # (B, 12, H, W)

        m = m.view(b, 3, 4, h_hr, w_hr)
        hr_aug = torch.cat([hr_base, torch.ones_like(hr_base[:, :1])], dim=1)
        delta = (m * hr_aug.unsqueeze(1)).sum(dim=2)
        out_coarse = hr_base + delta

        refine_in = torch.cat([out_coarse, hr, hr_edge], dim=1)
        refine = torch.tanh(self._stage(self.refine, refine_in)) * self.refine_bound
        return out_coarse + refine
