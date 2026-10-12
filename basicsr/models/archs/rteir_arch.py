"""RTEIR: Real-Time Event-guided Image Restoration network.

Architecture overview:
  * Half-resolution trunk via PixelUnshuffle / PixelShuffle.
  * Dual encoders + a single image-only decoder: event features never enter
    the decoder, only the conditioned image skips do.
  * Signed event stem (EVB) + four closed-form event moments, projected into
    a reliability condition pyramid shared by all stages.
  * Reliability-calibrated fusion (soft modal transport, CMF, BCA) at the
    shallow / mid encoder levels.
  * ViT bottleneck at the deepest level: windowed image self-attention +
    event-queried pooled cross-attention + gated FFN for both modalities,
    followed by direction-guided aggregation (DGA).

Self-contained (no external arch imports).

All operators are ONNX-exportable (no FFT, no dynamic control flow, no
Sinkhorn, no selective scan). Windowed attention uses fixed-size windows
with reflect-padding so that arbitrary input resolutions are supported at
train and test time.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_tuple(value, length, name="value"):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if len(value) == length and value.isdigit():
            parts = [int(v) for v in value]
        else:
            parts = [int(v.strip()) for v in value.split(",") if v.strip()]
        if len(parts) != length:
            raise ValueError(f"{name} must contain {length} values, got {value!r}")
        return tuple(parts)
    if isinstance(value, (list, tuple)):
        if len(value) != length:
            raise ValueError(f"{name} must contain {length} values, got {value!r}")
        return tuple(int(v) for v in value)
    raise TypeError(f"unsupported {name} type: {type(value)!r}")


def _pad_to_multiple(x, multiple=8):
    height, width = x.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, height, width
    pad_mode = "reflect" if height > 1 and width > 1 else "replicate"
    x = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)
    return x, height, width


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        eps_t = torch.tensor(self.eps, dtype=x.dtype, device=x.device)
        return (x - mean) * torch.rsqrt(var + eps_t) * self.weight + self.bias


class PreNorm(nn.Module):
    """LayerNorm applied over the channel dimension of a (B, N, C) tensor.

    Uses F.layer_norm directly for ONNX compatibility (avoids type-binding
    issues with nn.LayerNorm in some PyTorch versions).
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        return F.layer_norm(x, (self.dim,), self.weight, self.bias, self.eps)


class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, act=True, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                      padding=padding, groups=in_channels, bias=bias),
            nn.Conv2d(in_channels, out_channels, 1, bias=bias),
            nn.GELU() if act else nn.Identity(),
        )

    def forward(self, x):
        return self.body(x)


class GatedResBlock(nn.Module):
    """DW + H/V stripe + SimpleGate FFN."""

    def __init__(self, channels, expansion=2.5, kernel_size=5, stripe=7, layer_scale=1e-3):
        super().__init__()
        hidden = max(channels, int(round(channels * expansion)))
        pad = kernel_size // 2
        s_pad = stripe // 2
        self.norm = LayerNorm2d(channels)
        self.dw = nn.Conv2d(channels, channels, kernel_size, padding=pad, groups=channels, bias=True)
        self.stripe_h = nn.Conv2d(
            channels, channels, kernel_size=(1, stripe), padding=(0, s_pad), groups=channels, bias=True)
        self.stripe_v = nn.Conv2d(
            channels, channels, kernel_size=(stripe, 1), padding=(s_pad, 0), groups=channels, bias=True)
        self.ffn_in = nn.Conv2d(channels, hidden * 2, 1, bias=True)
        self.ffn_out = nn.Conv2d(hidden, channels, 1, bias=True)
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1) * layer_scale)

    def forward(self, x):
        y = self.norm(x)
        y = self.dw(y) + self.stripe_v(self.stripe_h(y))
        y1, y2 = self.ffn_in(y).chunk(2, dim=1)
        y = self.ffn_out(F.gelu(y1) * y2)
        return x + self.scale * y


# ---------------------------------------------------------------------------
# ViT components (ONNX-compatible)
# ---------------------------------------------------------------------------

class WindowPartition(nn.Module):
    """Partition spatial features into non-overlapping windows.

    Uses only reshape + permute -> fully ONNX-exportable.
    """

    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size

    def forward(self, x):
        """x: (B, C, H, W) -> windows: (B*num_win, window_size^2, C)"""
        B, C, H, W = x.shape
        ws = self.window_size
        x = x.reshape(B, C, H // ws, ws, W // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, C)
        return x


class WindowUnpartition(nn.Module):
    """Inverse of WindowPartition."""

    def __init__(self, window_size, H, W):
        super().__init__()
        self.window_size = window_size
        self.H = H
        self.W = W

    def forward(self, windows):
        """windows: (B*num_win, window_size^2, C) -> (B, C, H, W)"""
        ws = self.window_size
        B = windows.shape[0] // ((self.H // ws) * (self.W // ws))
        C = windows.shape[-1]
        nH, nW = self.H // ws, self.W // ws
        x = windows.reshape(B, nH, nW, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, self.H, self.W)
        return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias.

    All operators (Linear, reshape, softmax, matmul) are ONNX-exportable.
    """

    def __init__(self, dim, num_heads, window_size, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias table (learnable, ONNX-safe: just an embedding lookup)
        self.rel_bias = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Compute relative position index (registered as buffer, not a parameter)
        coords = torch.arange(window_size)
        relative_coords = coords.unsqueeze(0) - coords.unsqueeze(1)  # (ws, ws)
        relative_coords += window_size - 1  # shift to [0, 2*ws-2]
        # For 1D spatial (each window is ws*ws tokens flattened from ws x ws)
        # We use a simplified 1D relative position
        # Actually for 2D windows, we need 2D relative coords
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords_2d = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=0)  # (2, ws, ws)
        coords_flat = coords_2d.reshape(2, -1)  # (2, ws*ws)
        rel_coords = coords_flat.unsqueeze(2) - coords_flat.unsqueeze(1)  # (2, ws*ws, ws*ws)
        rel_coords = rel_coords.permute(1, 2, 0)  # (ws*ws, ws*ws, 2)
        rel_coords[:, :, 0] += window_size - 1
        rel_coords[:, :, 1] += window_size - 1
        rel_coords[:, :, 0] *= 2 * window_size - 1
        rel_index = rel_coords.sum(-1)  # (ws*ws, ws*ws)
        self.register_buffer("rel_index", rel_index, persistent=False)

    def forward(self, x):
        """x: (B*num_win, ws*ws, C) -> out: (B*num_win, ws*ws, C)"""
        Nw, N, C = x.shape
        H = self.num_heads
        D = self.head_dim

        qkv = self.qkv(x).reshape(Nw, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (Nw, H, N, D)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # Add relative position bias
        bias = self.rel_bias[self.rel_index.reshape(-1)].reshape(N, N, H)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)  # (1, H, N, N)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(Nw, N, C)
        return self.proj(out)


class PooledCrossAttention(nn.Module):
    """Cross-attention with adaptive-pool KV compression.

    Query tokens stay at full spatial resolution; key/value tokens are
    compressed to a small grid (kv_size x kv_size) via AdaptiveAvgPool2d.
    This keeps the attention matrix tiny (e.g. 33280 x 64) so the block
    is both fast and memory-efficient, while remaining fully ONNX-exportable.
    """

    def __init__(self, dim, num_heads, kv_size=8, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.kv_size = kv_size

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        # Strided average pool for KV compression (ONNX-safe, no shape deps).
        # stride=20 reduces 160->8 and 208->~10; the actual kv count is
        # determined at runtime and is small enough for efficient attention.
        self.kv_pool = nn.AvgPool2d(kernel_size=20, stride=20)

    def forward(self, query, context):
        """query, context: (B, C, H, W) -> (B, C, H, W)"""
        B, C, H, W = query.shape
        Hd = self.num_heads
        D = self.head_dim

        # Pool context to a small grid via fixed-stride avg pool (ONNX-safe)
        ctx = self.kv_pool(context)                      # (B, C, kh, kw)
        kv_len = ctx.shape[2] * ctx.shape[3]

        q = self.q_proj(query.permute(0, 2, 3, 1).reshape(B, H * W, C))
        q = q.reshape(B, H * W, Hd, D).permute(0, 2, 1, 3)          # (B, Hd, H*W, D)

        kv = self.kv_proj(ctx.permute(0, 2, 3, 1).reshape(B, kv_len, C))
        kv = kv.reshape(B, kv_len, 2, Hd, D).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)                                         # (B, Hd, kv_len, D)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        return self.out_proj(out).permute(0, 3, 1, 2)               # (B, C, H, W)


class ViTBlock(nn.Module):
    """Single ViT block: windowed self-attn (image) + pooled cross-attn (event)
    + gated FFN for both modalities.

    Input:  image (B, C, H, W), event (B, C, H, W)
            H, W must be divisible by window_size.
    Output: updated image, updated event  (same shapes)
    """

    def __init__(self, dim, num_heads, window_size, kv_size=8,
                 mlp_ratio=2.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)

        # --- Image self-attention path ---
        self.img_norm1 = PreNorm(dim)
        self.window_part = WindowPartition(window_size)
        self.w_attn = WindowAttention(dim, num_heads, window_size)
        self.img_norm2 = PreNorm(dim)
        self.img_ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )
        self.img_scale = nn.Parameter(torch.ones(dim) * 1e-3)

        # --- Event pooled cross-attention path ---
        self.evt_norm1 = PreNorm(dim)
        self.evt_norm2 = PreNorm(dim)
        self.cross_attn = PooledCrossAttention(dim, num_heads, kv_size=kv_size)
        self.cross_gate = nn.Parameter(torch.ones(dim) * 1e-3)
        self.evt_ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )
        self.evt_scale = nn.Parameter(torch.ones(dim) * 1e-3)

    def forward(self, img, evt):
        """img, evt: (B, C, H, W).  H, W already padded to ws multiple."""
        B, C, H, W = img.shape
        ws = self.w_attn.window_size

        # --- Image: windowed self-attention ---
        img_n = self.img_norm1(img.permute(0, 2, 3, 1))          # (B, H, W, C)
        windows = self.window_part(img_n.permute(0, 3, 1, 2))     # (B*nw, ws*ws, C)
        attn_out = self.w_attn(windows)                            # (B*nw, ws*ws, C)
        # Un-partition
        nH, nW = H // ws, W // ws
        attn_out = attn_out.reshape(B, nH, nW, ws, ws, C)
        attn_out = attn_out.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
        img = img + self.img_scale.view(1, C, 1, 1) * attn_out

        # --- Image: FFN ---
        img_ffn_in = self.img_norm2(img.permute(0, 2, 3, 1))      # (B, H, W, C)
        img = img + self.img_scale.view(1, C, 1, 1) * self.img_ffn(img_ffn_in).permute(0, 3, 1, 2)

        # --- Event: pooled cross-attention (event queries, pooled image KV) ---
        evt_n = self.evt_norm1(evt.permute(0, 2, 3, 1))           # (B, H, W, C)
        img_n2 = self.img_norm2(img.permute(0, 2, 3, 1))          # (B, H, W, C)
        cross_out = self.cross_attn(
            evt_n.permute(0, 3, 1, 2),                             # (B, C, H, W)
            img_n2.permute(0, 3, 1, 2),                            # (B, C, H, W)
        )                                                            # (B, C, H, W)
        evt = evt + self.cross_gate.view(1, C, 1, 1) * cross_out

        # --- Event: FFN ---
        evt_ffn_in = self.evt_norm2(evt.permute(0, 2, 3, 1))
        evt = evt + self.evt_scale.view(1, C, 1, 1) * self.evt_ffn(evt_ffn_in).permute(0, 3, 1, 2)

        return img, evt


class ViTBottleneck(nn.Module):
    """Full ViT bottleneck: pad -> N x ViT blocks -> crop.

    Handles arbitrary input sizes by reflect-padding to a multiple of
    window_size before attention, then cropping back.
    """

    def __init__(self, dim, num_heads, window_size, depth=3, kv_size=8,
                 mlp_ratio=2.0, drop=0.0):
        super().__init__()
        self.window_size = window_size
        self.blocks = nn.ModuleList([
            ViTBlock(dim, num_heads, window_size, kv_size=kv_size,
                     mlp_ratio=mlp_ratio, drop=drop)
            for _ in range(depth)
        ])
        self.img_norm = LayerNorm2d(dim)
        self.evt_norm = LayerNorm2d(dim)

    def forward(self, img, evt):
        """img, evt: (B, C, H, W) -> (B, C, H, W), same shape."""
        H, W = img.shape[-2:]
        ws = self.window_size

        # Pad to multiple of window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
            evt = F.pad(evt, (0, pad_w, 0, pad_h), mode="reflect")

        for blk in self.blocks:
            img, evt = blk(img, evt)

        # Crop back
        if pad_h > 0 or pad_w > 0:
            img = img[:, :, :H, :W]
            evt = evt[:, :, :H, :W]

        return self.img_norm(img), self.evt_norm(evt)


# ---------------------------------------------------------------------------
# Event processing modules
# ---------------------------------------------------------------------------

class EventMomentPyramid(nn.Module):
    """Voxel moments used as event reliability / temporal-direction priors."""

    def __init__(self, out_channels=16, num_bins=6, eps=1e-6):
        super().__init__()
        self.eps = float(eps)
        self.num_bins = num_bins
        self.project = nn.Sequential(
            nn.Conv2d(4, out_channels, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=True),
        )
        # Pre-compute bins as a fixed buffer (avoids torch.linspace -> double in ONNX)
        bins = torch.linspace(-1.0, 1.0, num_bins).view(1, num_bins, 1, 1)
        self.register_buffer("bins", bins, persistent=False)

    def _moments(self, event):
        bins = self.bins.to(dtype=event.dtype)
        abs_event = event.abs()
        activity = abs_event.mean(dim=1, keepdim=True)
        signed_balance = event.mean(dim=1, keepdim=True)
        # Use tensor eps to avoid Python scalar -> double in ONNX
        eps_t = torch.tensor(self.eps, dtype=event.dtype, device=event.device)
        mass = abs_event.sum(dim=1, keepdim=True) + eps_t
        centroid = (abs_event * bins).sum(dim=1, keepdim=True) / mass
        spread = (abs_event * (bins - centroid).pow(2)).sum(dim=1, keepdim=True) / mass
        return torch.cat([activity, signed_balance, centroid, spread], dim=1)

    def forward(self, event, target_sizes):
        event_half = F.avg_pool2d(event, kernel_size=2, stride=2)
        cond = self.project(self._moments(event_half))
        conds = []
        for size in target_sizes:
            if cond.shape[-2:] == size:
                conds.append(cond)
            else:
                conds.append(F.interpolate(cond, size=size, mode="bilinear", align_corners=False))
        return conds


class EventConditionAffine(nn.Module):
    def __init__(self, cond_channels, feat_channels):
        super().__init__()
        self.affine = nn.Sequential(
            nn.Conv2d(cond_channels, feat_channels, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(feat_channels, feat_channels * 2, 1, bias=True),
        )

    def forward(self, feat, cond):
        gamma, beta = self.affine(cond).chunk(2, dim=1)
        return feat * (1.0 + torch.tanh(gamma)) + beta


class SignedEventStem(nn.Module):
    """Separate positive/negative event evidence before first fusion (EVB)."""

    def __init__(self, in_channels, cond_channels, out_channels, expansion=2.0):
        super().__init__()
        mid = max(out_channels // 2, 16)
        self.pos = DSConv(in_channels, mid, kernel_size=3, act=True)
        self.neg = DSConv(in_channels, mid, kernel_size=3, act=True)
        self.merge = nn.Sequential(
            nn.Conv2d(mid * 2 + cond_channels, out_channels, 1, bias=True),
            nn.GELU(),
        )
        self.affine = EventConditionAffine(cond_channels, out_channels)
        self.body = GatedResBlock(out_channels, expansion=expansion, kernel_size=3, stripe=7)

    def forward(self, event_pack, cond):
        pos = self.pos(F.relu(event_pack))
        neg = self.neg(F.relu(-event_pack))
        feat = self.merge(torch.cat([pos, neg, cond], dim=1))
        return self.body(self.affine(feat, cond))


# ---------------------------------------------------------------------------
# Fusion modules
# ---------------------------------------------------------------------------

class SoftModalMixer(nn.Module):
    """MHC-lite: channel-wise 2x2 transport from pooled descriptors."""

    def __init__(self, channels, reduction=8, residual_alpha=0.5):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.residual_alpha = float(residual_alpha)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.logit = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels * 4, 1, bias=True),
        )

    def forward(self, img, event):
        b, c, _, _ = img.shape
        desc = torch.cat(
            [self.pool(img), self.pool(event), self.pool((img - event).abs())],
            dim=1,
        )
        mix = torch.softmax(self.logit(desc).view(b, 2, 2, c, 1, 1), dim=2)
        mixed_img = mix[:, 0, 0] * img + mix[:, 0, 1] * event
        mixed_event = mix[:, 1, 0] * img + mix[:, 1, 1] * event
        return (
            img + self.residual_alpha * (mixed_img - img),
            event + self.residual_alpha * (mixed_event - event),
        )


class CrossConvFusion(nn.Module):
    """Local cross-modal injection (CMF substitute)."""

    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True),
            nn.Conv2d(channels, channels, 1, bias=True),
        )
        self.gate = nn.Sequential(nn.Conv2d(channels * 2, channels, 1, bias=True), nn.Sigmoid())

    def forward(self, img, event):
        return img + self.gate(torch.cat([img, event], dim=1)) * self.proj(event)


class BCAFusion(nn.Module):
    """Bi-directional channel attention fusion."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)

        def _mlp():
            return nn.Sequential(
                nn.Conv2d(channels, hidden, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, channels, 1, bias=True),
                nn.Sigmoid(),
            )

        self.att_i = _mlp()
        self.att_e = _mlp()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, img, event):
        return img * self.att_i(self.pool(event)) + event * self.att_e(self.pool(img))


class ReliabilityLowFusion(nn.Module):
    def __init__(self, channels, cond_channels=16, reduction=8):
        super().__init__()
        self.event_gate = nn.Sequential(nn.Conv2d(cond_channels, channels, 1, bias=True), nn.Sigmoid())
        self.image_bias = nn.Conv2d(cond_channels, channels, 1, bias=True)
        self.mixer = SoftModalMixer(channels, reduction=reduction)
        self.cmf = CrossConvFusion(channels)
        self.event_out = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, img, event, cond):
        event = event * self.event_gate(cond)
        img = img + self.image_bias(cond)
        img, event = self.mixer(img, event)
        img = self.cmf(img, event)
        event = event + self.event_out(img - event)
        return img, event


class ReliabilityHighFusion(nn.Module):
    """Deep RGF: reliability gate + soft mixer + BCA -> feeds ViT bottleneck."""

    def __init__(self, channels, cond_channels=16, reduction=8):
        super().__init__()
        self.event_gate = nn.Sequential(nn.Conv2d(cond_channels, channels, 1, bias=True), nn.Sigmoid())
        self.image_bias = nn.Conv2d(cond_channels, channels, 1, bias=True)
        self.mixer = SoftModalMixer(channels, reduction=reduction)
        self.bca = BCAFusion(channels)

    def forward(self, img, event, cond):
        event = event * self.event_gate(cond)
        img = img + self.image_bias(cond)
        img, event = self.mixer(img, event)
        return img, self.bca(img, event)


class DirectionGuidedAggregation(nn.Module):
    """Five-direction aggregation guided by event support (DGA)."""

    def __init__(self, channels, cond_channels=16):
        super().__init__()
        self.weight = nn.Sequential(
            nn.Conv2d(channels + cond_channels, max(channels // 2, 16), 1, bias=True),
            nn.GELU(),
            nn.Conv2d(max(channels // 2, 16), 5, 1, bias=True),
        )
        self.proj = nn.Conv2d(channels, channels, 1, bias=True)
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1) * 1e-3)

    @staticmethod
    def _left(x):
        return F.pad(x[..., 1:], (0, 1, 0, 0))

    @staticmethod
    def _right(x):
        return F.pad(x[..., :-1], (1, 0, 0, 0))

    @staticmethod
    def _up(x):
        return F.pad(x[:, :, 1:, :], (0, 0, 0, 1))

    @staticmethod
    def _down(x):
        return F.pad(x[:, :, :-1, :], (0, 0, 1, 0))

    def forward(self, x, support, cond):
        weight = torch.softmax(self.weight(torch.cat([support, cond], dim=1)), dim=1)
        mixed = (
            weight[:, 0:1] * x
            + weight[:, 1:2] * self._left(x)
            + weight[:, 2:3] * self._right(x)
            + weight[:, 3:4] * self._up(x)
            + weight[:, 4:5] * self._down(x)
        )
        return x + self.scale * self.proj(mixed)


# ---------------------------------------------------------------------------
# Encoder / decoder stages
# ---------------------------------------------------------------------------

class StageDown(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=1, expansion=2.5):
        super().__init__()
        blocks = [DSConv(in_channels, out_channels, kernel_size=3, stride=2, act=True)]
        blocks += [GatedResBlock(out_channels, expansion=expansion) for _ in range(num_blocks)]
        self.body = nn.Sequential(*blocks)

    def forward(self, x):
        return self.body(x)


class StageUp(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, num_blocks=1, expansion=2.5):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 1, bias=True),
            nn.GELU(),
        )
        self.body = nn.Sequential(
            *[GatedResBlock(out_channels, expansion=expansion) for _ in range(num_blocks)]
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.body(self.proj(torch.cat([x, skip], dim=1)))


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------

class RTEIR(nn.Module):
    """Real-Time Event-guided Image Restoration network.

    Public interface:
        forward(x, event, mask=None) -> restored image
    """

    def __init__(
        self,
        inChannels_img=3,
        inChannels_event=6,
        outChannels=3,
        in_chn=None,
        ev_chn=None,
        out_chn=None,
        wf=48,
        channels=None,
        depths=None,
        bot_depth=3,
        cond_channels=16,
        expansion=2.5,
        vit_window_size=8,
        vit_num_heads=4,
        vit_kv_size=8,
        vit_mlp_ratio=2.0,
        ends_with_relu=False,
        return_intermediate=False,
        **kwargs,
    ):
        super().__init__()
        # Pop legacy kwargs for compatibility with shared training configs
        for key in (
            "ffn_expansion_factor", "depth", "use_egds", "use_bissm_fusion",
            "use_other_pack", "use_event_feedback", "d_state", "expand",
            "max_displacement", "mhc_reduction", "mhc_sinkhorn_iters",
            "mhc_temperature", "mhc_residual_alpha", "encoder_block_config",
            "img_encoder_block_config", "event_encoder_block_config",
            "use_evb", "use_rgf", "use_dga", "bot_blocks", "scf_alpha",
        ):
            if key == "ffn_expansion_factor" and key in kwargs:
                expansion = kwargs.pop(key)
            else:
                kwargs.pop(key, None)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected RTEIR arguments: {unknown}")

        if in_chn is not None:
            inChannels_img = in_chn
        if ev_chn is not None:
            inChannels_event = ev_chn
        if out_chn is not None:
            outChannels = out_chn

        channels = _to_tuple(channels, 3, "channels") or (
            int(wf),
            int(round(wf * 1.833)),
            int(round(wf * 4.0)),
        )
        depths = _to_tuple(depths, 3, "depths") or (1, 1, 2)
        c0, c1, c2 = channels
        d0, d1, d2 = depths

        self.in_channels_img = int(inChannels_img)
        self.in_channels_event = int(inChannels_event)
        self.out_channels = int(outChannels)
        self.ends_with_relu = bool(ends_with_relu)
        self.return_intermediate = bool(return_intermediate)

        img_pack_channels = self.in_channels_img * 4
        event_pack_channels = self.in_channels_event * 4

        # --- Condition pyramid ---
        self.cond_pyramid = EventMomentPyramid(
            out_channels=cond_channels, num_bins=self.in_channels_event)

        # --- Level 0: full half-res ---
        self.img_stem = nn.Sequential(
            DSConv(img_pack_channels + cond_channels, c0, kernel_size=3, act=True),
            *[GatedResBlock(c0, expansion=min(expansion, 2.0)) for _ in range(d0)],
        )
        self.event_stem = SignedEventStem(
            event_pack_channels, cond_channels, c0, expansion=min(expansion, 2.0)
        )
        self.fuse0 = ReliabilityLowFusion(c0, cond_channels=cond_channels)

        # --- Level 1: quarter-res ---
        self.img_down1 = StageDown(c0, c1, num_blocks=d1, expansion=expansion)
        self.event_down1 = StageDown(c0, c1, num_blocks=1, expansion=1.75)
        self.fuse1 = ReliabilityLowFusion(c1, cond_channels=cond_channels)

        # --- Level 2: eighth-res -> ViT bottleneck ---
        self.img_down2 = StageDown(c1, c2, num_blocks=d2, expansion=expansion)
        self.event_down2 = StageDown(c1, c2, num_blocks=1, expansion=1.75)
        self.fuse2 = ReliabilityHighFusion(c2, cond_channels=cond_channels)

        # ViT bottleneck
        self.vit_bot = ViTBottleneck(
            c2, num_heads=vit_num_heads, window_size=vit_window_size,
            depth=bot_depth, kv_size=vit_kv_size, mlp_ratio=vit_mlp_ratio,
        )
        self.dga_bot = DirectionGuidedAggregation(c2, cond_channels=cond_channels)

        # --- Decoder: plain StageUp cascade, no event fusion ---
        self.up1 = StageUp(c2, c1, c1, num_blocks=1, expansion=expansion)
        self.up0 = StageUp(c1, c0, c0, num_blocks=1, expansion=min(expansion, 2.0))

        # --- Output head ---
        self.out_head = nn.Sequential(
            LayerNorm2d(c0),
            DSConv(c0, c0, kernel_size=3, act=True),
            nn.Conv2d(c0, self.out_channels * 4, 3, padding=1, bias=True),
            nn.PixelShuffle(2),
        )

    def _pack_inputs(self, x, event):
        x_pad, height, width = _pad_to_multiple(x, multiple=8)
        if event.shape[-2:] != x.shape[-2:]:
            event = F.interpolate(event, size=x.shape[-2:], mode="bilinear", align_corners=False)
        event_pad, _, _ = _pad_to_multiple(event, multiple=8)
        return (
            x_pad, event, event_pad,
            F.pixel_unshuffle(x_pad, 2),
            F.pixel_unshuffle(event_pad, 2),
            height, width,
        )

    def forward(self, x, event, mask=None):
        del mask
        input_img, event_raw, event, img_pack, event_pack, height, width = self._pack_inputs(x, event)

        size0 = img_pack.shape[-2:]
        size1 = ((size0[0] + 1) // 2, (size0[1] + 1) // 2)
        size2 = ((size1[0] + 1) // 2, (size1[1] + 1) // 2)
        cond0, cond1, cond2 = self.cond_pyramid(event_raw, [size0, size1, size2])

        # Level 0
        img0 = self.img_stem(torch.cat([img_pack, cond0], dim=1))
        event0 = self.event_stem(event_pack, cond0)
        img0, event0 = self.fuse0(img0, event0, cond0)

        # Level 1
        img1 = self.img_down1(img0)
        event1 = self.event_down1(event0)
        img1, event1 = self.fuse1(img1, event1, cond1)

        # Level 2
        img2 = self.img_down2(img1)
        event2 = self.event_down2(event1)
        img2, event2 = self.fuse2(img2, event2, cond2)

        # ViT bottleneck
        img2, event2 = self.vit_bot(img2, event2)
        img2 = self.dga_bot(img2, event2, cond2)

        # Decoder: plain StageUp cascade (no event fusion)
        dec1 = self.up1(img2, img1)
        dec0 = self.up0(dec1, img0)

        out = self.out_head(dec0) + input_img
        out = out[:, :, :height, :width]
        if self.ends_with_relu:
            out = torch.relu(out)
        if self.return_intermediate:
            return [out]
        return out


# ---------------------------------------------------------------------------
# Registration (auto-discovered by basicsr/models/archs/__init__.py)
# ---------------------------------------------------------------------------

globals()["RTEIR"] = RTEIR
globals()["rteir"] = RTEIR

