#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
model.py  ―  FaultTrans

Belief/commit model with connectivity enforced only at the committed/final layer.

Semantics:
- belief    : current-frame hypothesis, can go up or down, may be exploratory
- committed : stable + connected subset of belief, monotonic over time
- final     : committed  (recommended for visualisation / inference)

Changes vs. previous version
─────────────────────────────
1. Model class renamed  FaultCleanRule → FaultTrans
   (FaultCleanRule / FaultSegModel kept as aliases for backward-compat)

2. FaultLossClean._tv_loss now takes `strike` and implements
   **Anisotropic TV**:
     • 0.1 × along-strike gradient²   (allow elongated fault to grow)
     • 0.9 × across-strike gradient²  (suppress off-axis fragments)

3. FaultLossClean.__init__ adds two new hyper-parameters:
     • pos_weight (float, default 3.0) – BCE positive-class weight for
       sparse fault pixels (~5 % of grid).
     • min_ecc    (float, default 0.01) – minimum eccentricity gate in
       _direction_loss; frames whose predicted mask is nearly circular
       are excluded, preventing NaN/Inf gradients.

4. _bce_from_logits uses pos_weight.

5. _direction_loss gates on both mass AND eccentricity (norm.detach()).

Run a smoke test:
    python model.py --smoke_test
"""

from typing import Optional, Tuple
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

GRID_SIZE = 150


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_causal_mask(T: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)


# ─────────────────────────────────────────────────────────────────────────────
# Patch embedding
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels: int = 1, patch_size: int = 10, d_model: int = 128):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels,
            d_model,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        B, D, T, Hp, Wp = x.shape
        return x.permute(0, 2, 3, 4, 1).reshape(B, T, Hp * Wp, D)


# ─────────────────────────────────────────────────────────────────────────────
# Physics positional encoding
# ─────────────────────────────────────────────────────────────────────────────

class PhysicsPositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        grid_size: int = 150,
        patch_size: int = 10,
        v_s: float = 3.5,
        resolution: float = 0.01,
        t_total: float = 40.0,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.n_patch   = grid_size // patch_size
        self.v_s       = v_s
        self.res_km    = resolution * 111.0
        self.t_total   = t_total
        self.proj      = nn.Linear(5, d_model)

    def forward(
        self,
        l_spatial: int,
        times: torch.Tensor,
        epicenter: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T   = times.shape
        device = times.device
        idx    = torch.arange(l_spatial, device=device)

        px = ((idx % self.n_patch).float() + 0.5) / self.n_patch
        py = ((idx // self.n_patch).float() + 0.5) / self.n_patch

        if epicenter is None:
            epi = torch.full((B, 2), 0.5, device=device)
        else:
            epi = epicenter.float()
            if epi.dim() > 2:
                epi = epi[:, 0, :]
            epi = epi.clamp(0.0, 1.0)

        ex = epi[:, 0:1]
        ey = epi[:, 1:2]

        dx        = px.unsqueeze(0) - ex
        dy        = py.unsqueeze(0) - ey
        dist_norm = torch.sqrt(dx.pow(2) + dy.pow(2)).clamp(min=0.0)
        dist_pix  = dist_norm * (self.grid_size - 1)
        dist_km   = dist_pix * self.res_km

        x_norm    = px.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        y_norm    = py.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        t_norm    = times.unsqueeze(-1).expand(B, T, l_spatial) / self.t_total
        t_shift   = (times.unsqueeze(-1) - dist_km.unsqueeze(1) / self.v_s) / self.t_total
        dist_norm = dist_norm.unsqueeze(1).expand(B, T, -1)

        feat = torch.stack([x_norm, y_norm, t_norm, t_shift, dist_norm], dim=-1)
        return self.proj(feat)


# ─────────────────────────────────────────────────────────────────────────────
# Spatio-temporal transformer block
# ─────────────────────────────────────────────────────────────────────────────

class SSMBlock(nn.Module):
    def __init__(
        self,
        d_model:   int,
        num_heads: int   = 4,
        expand:    int   = 2,
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.norm_s = nn.LayerNorm(d_model)
        self.attn_s = nn.MultiheadAttention(
            d_model, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm_t = nn.LayerNorm(d_model)
        self.attn_t = nn.MultiheadAttention(
            d_model, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm2  = nn.LayerNorm(d_model)
        self.ffn    = nn.Sequential(
            nn.Linear(d_model, d_model * expand),
            nn.GELU(),
            nn.Linear(d_model * expand, d_model),
        )

    def _spatial_attn(self, x: torch.Tensor) -> torch.Tensor:
        B, T, L, D = x.shape
        xt  = x.reshape(B * T, L, D)
        xn  = self.norm_s(xt)
        out, _ = self.attn_s(xn, xn, xn, need_weights=False)
        return (xt + out).reshape(B, T, L, D)

    def _temporal_attn(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, T, L, D = x.shape
        xt  = x.permute(0, 2, 1, 3).reshape(B * L, T, D)
        xn  = self.norm_t(xt)
        out, _ = self.attn_t(xn, xn, xn, attn_mask=causal_mask, need_weights=False)
        return (xt + out).reshape(B, L, T, D).permute(0, 2, 1, 3)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        seq_mask:    Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self._spatial_attn(x)
        x = self._temporal_attn(x, causal_mask)

        if seq_mask is not None:
            x = x * seq_mask.float()[:, :, None, None]

        B, T, L, D = x.shape
        xf = x.reshape(B * T * L, D)
        xf = xf + self.ffn(self.norm2(xf))
        x  = xf.reshape(B, T, L, D)

        if seq_mask is not None:
            x = x * seq_mask.float()[:, :, None, None]
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Decoder components
# ─────────────────────────────────────────────────────────────────────────────

class CoordInjection(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        device     = x.device
        ys = torch.linspace(-1.0, 1.0, H, device=device)
        xs = torch.linspace(-1.0, 1.0, W, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        rr     = torch.sqrt(gx * gx + gy * gy).clamp(max=1.4143)
        coord  = torch.stack([gx, gy, rr], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        return torch.cat([x, coord], dim=1)


class FiLM2D(nn.Module):
    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.to_gamma_beta = nn.Sequential(
            nn.Linear(cond_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim * 2),
        )

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma_beta        = self.to_gamma_beta(cond)
        gamma, beta       = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return feat * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta


class ResDWBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups    = 8 if channels % 8 == 0 else 4 if channels % 4 == 0 else 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.dw    = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pw    = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.ffn   = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pw(self.dw(self.norm1(x)))
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class PixelShuffleUp(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, upscale: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * upscale * upscale, kernel_size=3, padding=1),
            nn.GELU(),
            nn.PixelShuffle(upscale),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class AxialAttention2D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.row_norm = nn.LayerNorm(channels)
        self.row_attn = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.col_norm = nn.LayerNorm(channels)
        self.col_attn = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn      = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # row attention
        xr   = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
        xr_n = self.row_norm(xr)
        xr   = xr + self.row_attn(xr_n, xr_n, xr_n, need_weights=False)[0]
        xr   = xr + self.ffn(self.ffn_norm(xr))
        x    = xr.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # column attention
        xc   = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
        xc_n = self.col_norm(xc)
        xc   = xc + self.col_attn(xc_n, xc_n, xc_n, need_weights=False)[0]
        xc   = xc + self.ffn(self.ffn_norm(xc))
        x    = xc.reshape(B, W, H, C).permute(0, 3, 2, 1)
        return x


class HybridFeatureDecoder(nn.Module):
    def __init__(
        self,
        d_model:          int,
        patch_size:       int,
        n_patches_side:   int,
        out_dim:          int = 32,
        mid_dim:          int = 64,
        num_refine_blocks: int = 3,
    ):
        super().__init__()
        self.patch_size     = patch_size
        self.n_patches_side = n_patches_side
        self.out_dim        = out_dim
        self.mid_dim        = mid_dim

        self.token_proj = nn.Linear(d_model, mid_dim)
        self.coord      = CoordInjection()

        self.stem = nn.Sequential(
            nn.Conv2d(mid_dim + 3, mid_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.film0 = FiLM2D(d_model, mid_dim)

        self.up2        = PixelShuffleUp(mid_dim, mid_dim, upscale=2)
        self.axial      = AxialAttention2D(mid_dim, num_heads=4, dropout=0.0)
        self.mid_refine = ResDWBlock(mid_dim)
        self.film1      = FiLM2D(d_model, mid_dim)

        self.full_refine = nn.Sequential(
            *[ResDWBlock(mid_dim) for _ in range(num_refine_blocks)]
        )
        self.film2 = FiLM2D(d_model, mid_dim)

        self.out_proj = nn.Sequential(
            nn.Conv2d(mid_dim + 3, mid_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_dim, out_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, L, D = x.shape
        n   = self.n_patches_side
        H   = W = n * self.patch_size

        g = x.mean(dim=2)
        x = self.token_proj(x)
        x = x.reshape(B * T, n, n, self.mid_dim).permute(0, 3, 1, 2)
        g = g.reshape(B * T, D)

        x = self.coord(x)
        x = self.stem(x)
        x = self.film0(x, g)

        x = self.up2(x)
        x = self.axial(x)
        x = self.mid_refine(x)
        x = self.film1(x, g)

        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        x = self.full_refine(x)
        x = self.film2(x, g)

        x = self.coord(x)
        x = self.out_proj(x)
        return x.permute(0, 2, 3, 1).reshape(B, T, H, W, self.out_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Recurrent state modules
# ─────────────────────────────────────────────────────────────────────────────

class StateFusion(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.state_conv = nn.Conv2d(2, feat_dim, kernel_size=3, padding=1)
        self.fuse       = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(
        self,
        feat_t:        torch.Tensor,
        belief_prev:   torch.Tensor,
        committed_prev: torch.Tensor,
    ) -> torch.Tensor:
        state      = torch.stack([belief_prev, committed_prev], dim=1)
        state_feat = self.state_conv(state).permute(0, 2, 3, 1)
        fused      = torch.cat([feat_t, state_feat], dim=-1)
        return self.fuse(fused)


class BeliefHead(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Linear(feat_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CommitHead(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Linear(feat_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Main model  ―  FaultTrans
# ─────────────────────────────────────────────────────────────────────────────

class FaultTrans(nn.Module):
    """
    FaultTrans: real-time fault-rupture spatial segmentation model.

    Input  : incremental PGV (or PGA / intensity) field  [B, 1, T, H, W]
    Output : (belief_logits [B, T, H, W],
              committed     [B, T, H, W])
    """

    def __init__(
        self,
        in_channels:          int   = 1,
        patch_size:           int   = 10,
        d_model:              int   = 256,
        n_layers:             int   = 6,
        grid_size:            int   = 150,
        decoder_dim:          int   = 32,
        decoder_mid_dim:      int   = 64,
        decoder_refine_blocks: int  = 3,
        v_s:                  float = 3.5,
        resolution:           float = 0.01,
        t_total:              float = 40.0,
        causal:               bool  = True,
        commit_tau:           float = 0.60,
        commit_temp:          float = 0.08,
        conn_tau:             float = 0.05,
        conn_temp:            float = 0.05,
        conn_radius:          int   = 3,
        epicenter_sigma:      float = 0.02,
        epicenter_scale:      float = 1.0,
    ):
        super().__init__()
        assert grid_size % patch_size == 0

        n_patches_side       = grid_size // patch_size
        self.grid_size       = grid_size
        self.causal          = causal
        self.commit_tau      = commit_tau
        self.commit_temp     = commit_temp
        self.conn_tau        = conn_tau
        self.conn_temp       = conn_temp
        self.conn_radius     = conn_radius
        self.epicenter_sigma = epicenter_sigma
        self.epicenter_scale = epicenter_scale

        self.patch_embed = PatchEmbed3D(
            in_channels=in_channels,
            patch_size=patch_size,
            d_model=d_model,
        )
        self.physics_pe = PhysicsPositionalEncoding(
            d_model=d_model,
            grid_size=grid_size,
            patch_size=patch_size,
            v_s=v_s,
            resolution=resolution,
            t_total=t_total,
        )
        self.blocks = nn.ModuleList(
            [SSMBlock(d_model=d_model) for _ in range(n_layers)]
        )
        self.feature_decoder = HybridFeatureDecoder(
            d_model=d_model,
            patch_size=patch_size,
            n_patches_side=n_patches_side,
            out_dim=decoder_dim,
            mid_dim=decoder_mid_dim,
            num_refine_blocks=decoder_refine_blocks,
        )
        self.state_fusion = StateFusion(decoder_dim)
        self.belief_head  = BeliefHead(decoder_dim)
        self.commit_head  = CommitHead(decoder_dim)

    # ── connectivity helpers ──────────────────────────────────────────────────

    @staticmethod
    def soft_dilate(x: torch.Tensor, radius: int = 3) -> torch.Tensor:
        if radius <= 1:
            return x
        if radius % 2 == 0:
            radius += 1
        pad = radius // 2
        return F.max_pool2d(
            x.unsqueeze(1), kernel_size=radius, stride=1, padding=pad
        ).squeeze(1)

    def _make_epicenter_seed(
        self,
        epicenter: Optional[torch.Tensor],
        B: int,
        H: int,
        W: int,
        device: torch.device,
    ) -> torch.Tensor:
        if epicenter is None:
            epi = torch.full((B, 2), 0.5, device=device)
        else:
            epi = epicenter.float()
            if epi.dim() > 2:
                epi = epi[:, 0, :]
            if epi.abs().max() > 1.0:
                epi = epi / (self.grid_size - 1)
            epi = epi.clamp(0.0, 1.0)

        ys = torch.linspace(0.0, 1.0, H, device=device)
        xs = torch.linspace(0.0, 1.0, W, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        gx = gx.unsqueeze(0)
        gy = gy.unsqueeze(0)

        cx   = epi[:, 0].view(B, 1, 1)
        cy   = epi[:, 1].view(B, 1, 1)
        dist2 = (gx - cx).pow(2) + (gy - cy).pow(2)
        seed  = self.epicenter_scale * torch.exp(
            -dist2 / (2 * self.epicenter_sigma ** 2)
        )
        return seed.clamp(0.0, 1.0)

    # ── forward sub-steps ────────────────────────────────────────────────────

    def _extract_features(
        self,
        inputs:    torch.Tensor,
        times:     torch.Tensor,
        seq_mask:  Optional[torch.Tensor],
        epicenter: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = self.patch_embed(inputs)
        x = x + self.physics_pe(x.shape[2], times, epicenter)

        if seq_mask is not None:
            x = x * seq_mask.float()[:, :, None, None]

        causal_mask = (
            _build_causal_mask(x.shape[1], x.device) if self.causal else None
        )
        for block in self.blocks:
            x = block(x, causal_mask=causal_mask, seq_mask=seq_mask)

        return self.feature_decoder(x)

    def _decode_sequence(
        self,
        features:  torch.Tensor,
        seq_mask:  Optional[torch.Tensor],
        epicenter: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, H, W, _ = features.shape
        device = features.device

        belief_prev    = torch.zeros(B, H, W, device=device)
        committed_prev = torch.zeros(B, H, W, device=device)
        epicenter_seed = self._make_epicenter_seed(epicenter, B, H, W, device)

        belief_logits_all = []
        committeds_all    = []

        for t in range(T):
            feat_t = features[:, t]
            fused  = self.state_fusion(feat_t, belief_prev, committed_prev)

            belief_logits_t  = self.belief_head(fused)
            belief_prob_t    = torch.sigmoid(belief_logits_t)
            commit_gate_t    = torch.sigmoid(self.commit_head(fused))

            stable_t = torch.minimum(belief_prev, belief_prob_t)

            reachable_t = torch.maximum(
                self.soft_dilate(committed_prev, radius=self.conn_radius),
                epicenter_seed,
            )
            connectivity_gate_t = torch.sigmoid(
                (reachable_t - self.conn_tau) / self.conn_temp
            )

            lock_support_t = stable_t * commit_gate_t
            locked_t       = (
                torch.sigmoid((lock_support_t - self.commit_tau) / self.commit_temp)
                * stable_t
            )
            locked_t   = locked_t * connectivity_gate_t
            committed_t = torch.maximum(committed_prev, locked_t)

            if seq_mask is not None:
                valid_t       = seq_mask[:, t].float().view(B, 1, 1)
                belief_prob_t = belief_prob_t * valid_t + belief_prev    * (1.0 - valid_t)
                committed_t   = committed_t   * valid_t + committed_prev * (1.0 - valid_t)

            belief_logits_all.append(belief_logits_t)
            committeds_all.append(committed_t)

            belief_prev    = belief_prob_t
            committed_prev = committed_t

        belief_logits = torch.stack(belief_logits_all, dim=1)
        committeds    = torch.stack(committeds_all,    dim=1)
        return belief_logits, committeds

    # ── main forward ─────────────────────────────────────────────────────────

    def forward(
        self,
        inputs:    torch.Tensor,
        times:     torch.Tensor,
        seq_mask:  Optional[torch.Tensor] = None,
        epicenter: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if epicenter is not None:
            epi = epicenter.float()
            if epi.abs().max() > 1.0:
                epi = epi / (self.grid_size - 1)
            epicenter = epi.clamp(0.0, 1.0)

        features = self._extract_features(inputs, times, seq_mask, epicenter)
        return self._decode_sequence(features, seq_mask, epicenter)


# Backward-compatibility aliases
FaultCleanRule = FaultTrans
FaultSegModel  = FaultTrans


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class FaultLossClean(nn.Module):
    """
    Combined loss for FaultTrans:

      loss_belief    = dice + bce   instant GT supervision  (belief)
      loss_committed = dice + bce   cumulative GT supervision (committed)
      tv_loss        = anisotropic TV   suppress off-axis fragments while
                       allowing elongated fault traces to grow freely
      dir_loss       = soft inertia-tensor direction loss

    Anisotropic TV
    ──────────────
    Given the fault strike angle θ, spatial gradients are decomposed into:
      • along-strike  component  → weight α  (0.1 by default, permissive)
      • across-strike component  → weight β  (0.9 by default, suppressive)

    This lets thin elongated faults extend without TV penalising their edges,
    while still suppressing off-axis debris fragments.

    Direction loss  (soft inertia-tensor method)
    ─────────────────────────────────────────────
    Treats committed probability as "mass" and computes the inertia tensor:
        μ₂₀ = Σ p(x-cx)²,  μ₀₂ = Σ p(y-cy)²,  μ₁₁ = Σ p(x-cx)(y-cy)
    Principal-axis vector: pred_axis = (μ₂₀-μ₀₂, 2μ₁₁)
    Reference vector:       ref_axis = (cos 2θ, sin 2θ)
    loss = 1 - dot(pred, ref) / ‖pred‖

    Numerical stability:
      • norm = √(ax²+ay²+ε),  ε = 1e-6
      • frames with mass < min_mass  are excluded  (no signal)
      • frames with norm < min_ecc   are excluded  (near-circular → NaN risk)

    New hyper-parameters vs. previous version
    ──────────────────────────────────────────
      pos_weight  (default 3.0)  BCE positive-class weight for sparse pixels
      min_ecc     (default 0.01) eccentricity gate in direction loss
      tv_along_w  (default 0.1)  TV weight along strike direction
      tv_perp_w   (default 0.9)  TV weight across strike direction
    """

    def __init__(
        self,
        belief_w:    float = 0.4,
        committed_w: float = 0.4,
        tv_w:        float = 0.15,
        dir_w:       float = 0.05,
        dice_w:      float = 0.6,
        bce_w:       float = 0.4,
        pos_weight:  float = 3.0,    # BCE positive-class weight
        min_mass:    float = 1.0,    # direction loss: min committed mass
        min_ecc:     float = 0.01,   # direction loss: min eccentricity (NaN guard)
        tv_along_w:  float = 0.1,    # anisotropic TV: along-strike weight
        tv_perp_w:   float = 0.9,    # anisotropic TV: across-strike weight
        eps:         float = 1e-6,
    ):
        super().__init__()
        self.belief_w    = belief_w
        self.committed_w = committed_w
        self.tv_w        = tv_w
        self.dir_w       = dir_w
        self.dice_w      = dice_w
        self.bce_w       = bce_w
        self.pos_weight  = pos_weight
        self.min_mass    = min_mass
        self.min_ecc     = min_ecc
        self.tv_along_w  = tv_along_w
        self.tv_perp_w   = tv_perp_w
        self.eps         = eps

    # ── pixel-level losses ────────────────────────────────────────────────────

    def _dice_from_logits(self, logits, target, weight):
        prob  = torch.sigmoid(logits).flatten(2)
        tgt   = target.flatten(2)
        inter = (prob * tgt).sum(dim=2)
        denom = prob.sum(dim=2) + tgt.sum(dim=2)
        loss  = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
        return (loss * weight).sum() / (weight.sum() + self.eps)

    def _bce_from_logits(self, logits, target, weight):
        """BCE with positive-class weight to handle sparse fault pixels."""
        pw  = torch.tensor(self.pos_weight, device=logits.device)
        bce = F.binary_cross_entropy_with_logits(
            logits, target,
            pos_weight=pw,
            reduction="none",
        ).mean(dim=(-2, -1))
        return (bce * weight).sum() / (weight.sum() + self.eps)

    def _dice_from_prob(self, prob, target, weight):
        p     = prob.flatten(2)
        tgt   = target.flatten(2)
        inter = (p * tgt).sum(dim=2)
        denom = p.sum(dim=2) + tgt.sum(dim=2)
        loss  = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
        return (loss * weight).sum() / (weight.sum() + self.eps)

    def _bce_from_prob(self, prob, target, weight):
        prob   = prob.clamp(1e-4, 1.0 - 1e-4)
        logits = torch.log(prob) - torch.log1p(-prob)
        pw     = torch.tensor(self.pos_weight, device=logits.device)
        bce    = F.binary_cross_entropy_with_logits(
            logits, target,
            pos_weight=pw,
            reduction="none",
        ).mean(dim=(-2, -1))
        return (bce * weight).sum() / (weight.sum() + self.eps)

    # ── anisotropic TV loss ───────────────────────────────────────────────────

    def _tv_loss(
        self,
        prob:   torch.Tensor,   # [B, T, H, W]
        strike: torch.Tensor,   # [B]  fault strike in radians
        weight: torch.Tensor,   # [B, T]
    ) -> torch.Tensor:
        """
        Anisotropic Total Variation.

        Spatial gradients are projected onto the along-strike and
        across-strike axes.  The across-strike component is penalised
        heavily (tv_perp_w) to suppress off-axis fragments, while the
        along-strike component receives a small penalty (tv_along_w) so
        that thin elongated fault traces can grow freely.

        Projection:
          gx_along =  cos(θ) · gx     (x-gradient along strike)
          gx_perp  =  sin(θ) · gx     (x-gradient across strike)
          gy_along =  sin(θ) · gy
          gy_perp  =  cos(θ) · gy
        """
        B = prob.shape[0]

        gx = prob[:, :, :, 1:] - prob[:, :, :, :-1]   # [B,T,H,W-1]
        gy = prob[:, :, 1:, :] - prob[:, :, :-1, :]   # [B,T,H-1,W]

        cos_s = torch.cos(strike).view(B, 1, 1, 1)
        sin_s = torch.sin(strike).view(B, 1, 1, 1)

        gx_along = cos_s * gx
        gx_perp  = sin_s * gx
        gy_along = sin_s * gy
        gy_perp  = cos_s * gy

        loss_along = (
            gx_along.pow(2).mean(dim=(-2, -1))
            + gy_along.pow(2).mean(dim=(-2, -1))
        )
        loss_perp = (
            gx_perp.pow(2).mean(dim=(-2, -1))
            + gy_perp.pow(2).mean(dim=(-2, -1))
        )

        loss = self.tv_along_w * loss_along + self.tv_perp_w * loss_perp
        return (loss * weight).sum() / (weight.sum() + self.eps)

    # ── direction loss ────────────────────────────────────────────────────────

    def _direction_loss(
        self,
        prob:   torch.Tensor,   # [B, T, H, W]
        strike: torch.Tensor,   # [B]
        weight: torch.Tensor,   # [B, T]
    ) -> torch.Tensor:
        """
        Soft inertia-tensor direction loss.

        Frames are excluded when:
          • mass  < min_mass  (too few activated pixels – no signal)
          • norm  < min_ecc   (near-circular distribution – NaN/Inf risk)
        """
        B, T, H, W = prob.shape
        device = prob.device

        ys = torch.linspace(-1.0, 1.0, H, device=device)
        xs = torch.linspace(-1.0, 1.0, W, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")   # [H, W]
        gx = gx.unsqueeze(0).unsqueeze(0)                 # [1, 1, H, W]
        gy = gy.unsqueeze(0).unsqueeze(0)

        # ── mass and centroid ────────────────────────────────────────────────
        mass = prob.sum(dim=(-2, -1)).clamp_min(self.eps)   # [B, T]
        cx   = (prob * gx).sum(dim=(-2, -1)) / mass
        cy   = (prob * gy).sum(dim=(-2, -1)) / mass

        dx = gx - cx.unsqueeze(-1).unsqueeze(-1)
        dy = gy - cy.unsqueeze(-1).unsqueeze(-1)

        # ── second-order moments ─────────────────────────────────────────────
        mu20 = (prob * dx * dx).sum(dim=(-2, -1)) / mass
        mu02 = (prob * dy * dy).sum(dim=(-2, -1)) / mass
        mu11 = (prob * dx * dy).sum(dim=(-2, -1)) / mass

        # ── predicted principal-axis vector ──────────────────────────────────
        ax   = mu20 - mu02           # [B, T]
        ay   = 2.0 * mu11

        # ── reference vector  (2θ removes 180° ambiguity) ───────────────────
        ref_x = torch.cos(2.0 * strike).unsqueeze(1).expand_as(ax)
        ref_y = torch.sin(2.0 * strike).unsqueeze(1).expand_as(ay)

        dot  = ax * ref_x + ay * ref_y
        norm = torch.sqrt(ax * ax + ay * ay + self.eps)   # ≥ sqrt(ε) = 1e-3

        loss = 1.0 - dot / norm

        # ── gate: exclude low-mass AND near-circular frames ──────────────────
        valid = (
            weight
            * (mass >= self.min_mass).float()
            * (norm.detach() >= self.min_ecc).float()   # detach: gate only
        )

        if valid.sum() < 1:
            return prob.new_tensor(0.0)

        return (loss * valid).sum() / (valid.sum() + self.eps)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        belief_logits: torch.Tensor,            # [B, T, H, W]
        committeds:    torch.Tensor,            # [B, T, H, W]
        target:        torch.Tensor,            # [B, T, H, W]
        valid_frame:   torch.Tensor,            # [B, T]
        seq_mask:      torch.Tensor,            # [B, T]
        strike:        Optional[torch.Tensor] = None,   # [B]
    ) -> torch.Tensor:
        weight        = valid_frame * seq_mask.float()
        instant_gt    = target
        cumulative_gt = torch.cummax(target, dim=1)[0]

        loss_belief = (
            self.dice_w * self._dice_from_logits(belief_logits, instant_gt, weight)
            + self.bce_w  * self._bce_from_logits(belief_logits, instant_gt, weight)
        )
        loss_committed = (
            self.dice_w * self._dice_from_prob(committeds, cumulative_gt, weight)
            + self.bce_w  * self._bce_from_prob(committeds, cumulative_gt, weight)
        )

        # anisotropic TV: requires strike; fall back to isotropic if missing
        if strike is not None:
            tv = self._tv_loss(committeds, strike, weight)
        else:
            gx   = committeds[:, :, :, 1:] - committeds[:, :, :, :-1]
            gy   = committeds[:, :, 1:, :] - committeds[:, :, :-1, :]
            loss = gx.pow(2).mean(dim=(-2, -1)) + gy.pow(2).mean(dim=(-2, -1))
            tv   = (loss * weight).sum() / (weight.sum() + self.eps)

        total = (
            self.belief_w    * loss_belief
            + self.committed_w * loss_committed
            + self.tv_w        * tv
        )

        if self.dir_w > 0 and strike is not None:
            # Apply direction loss to both belief and committed.
            # Belief receives a higher weight (0.7) because its gradients are
            # unconstrained and can freely reshape the mask orientation.
            # Committed receives a smaller weight (0.3) as a stabilising anchor.
            belief_prob = torch.sigmoid(belief_logits)
            dir_loss = (
                0.7 * self._direction_loss(belief_prob, strike, weight)
                + 0.3 * self._direction_loss(committeds, strike, weight)
            )
            total = total + self.dir_w * dir_loss

        return total


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "FaultTrans",
    "FaultCleanRule",   # alias
    "FaultSegModel",    # alias
    "FaultLossClean",
    "HybridFeatureDecoder",
]


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def _smoke_test() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = FaultTrans(
        in_channels=1,
        patch_size=10,
        d_model=128,
        n_layers=2,
        grid_size=150,
        decoder_dim=16,
        decoder_mid_dim=32,
        decoder_refine_blocks=1,
    ).to(device)

    criterion = FaultLossClean()

    B, T, H, W = 2, 8, 150, 150
    inputs    = torch.randn(B, 1, T, H, W, device=device)
    times     = torch.linspace(0.0, 40.0, T, device=device).unsqueeze(0).repeat(B, 1)
    seq_mask  = torch.ones(B, T, dtype=torch.bool, device=device)
    epicenter = torch.rand(B, 2, device=device)
    strike    = torch.rand(B, device=device) * 3.14159
    target    = (torch.rand(B, T, H, W, device=device) > 0.9).float()
    valid_frame = torch.ones(B, T, device=device)

    belief_logits, committeds = model(inputs, times, seq_mask, epicenter)

    print("belief_logits :", tuple(belief_logits.shape))
    print("committeds    :", tuple(committeds.shape))

    loss = criterion(
        belief_logits=belief_logits,
        committeds=committeds,
        target=target,
        valid_frame=valid_frame,
        seq_mask=seq_mask,
        strike=strike,
    )
    print(f"loss          : {loss.item():.4f}")
    print("smoke test passed ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        _smoke_test()
    else:
        print(
            "This file defines the FaultTrans model.\n"
            "Run  python model.py --smoke_test  for a quick forward pass."
        )