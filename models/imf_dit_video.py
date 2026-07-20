"""PyTorch video imfDiT: port of models/imfDiT.py (JAX/Flax) extended from
images to video latents.

Input latents: (B, C, T, H, W) with C = 16 (Wan2.1 VAE latent channels).
3D patchify with patch size (1, 2, 2) -> tokens = T * (H/2) * (W/2).

Every op is torch.func.jvp-compatible (forward-mode AD): no in-place ops on
autograd tensors, no .item() in forward.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


#################################################################################
#                          Pluggable Attention Kernels                           #
#################################################################################


def sdpa_math_attention(q, k, v):
    """Default attention: torch SDPA restricted to the math backend.

    The math backend is a plain softmax(q k^T / sqrt(d)) v decomposition and
    therefore supports forward-mode AD (torch.func.jvp). Fused flash /
    mem-efficient kernels do not; a CuTeDSL flash-attention JVP op can be
    plugged in later via the `attn_impl` constructor argument.

    Args:
        q, k, v: (batch, seq_len, num_heads, head_dim)

    Returns:
        (batch, seq_len, num_heads, head_dim)
    """
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))  # -> (B, H, S, D)
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2)


#################################################################################
#                              Basic Torch Modules                               #
#################################################################################


def scaled_variance_linear(in_features, out_features, bias, init_constant):
    """Linear layer with weight ~ Normal(0, init_constant / sqrt(in_features))."""
    linear = nn.Linear(in_features, out_features, bias=bias)
    nn.init.normal_(linear.weight, std=init_constant / math.sqrt(in_features))
    if bias:
        nn.init.zeros_(linear.bias)
    return linear


class RMSNorm(nn.Module):
    """Root Mean Square Normalization (float32 accumulation)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_f32 = x.float()
        mean_square = x_f32.pow(2).mean(dim=-1, keepdim=True)
        normed = x_f32 * torch.rsqrt(mean_square + self.eps)
        return normed.to(x.dtype) * self.weight


class SwiGLUMlp(nn.Module):
    """Swish-Gated Linear Unit MLP."""

    def __init__(self, in_features, hidden_features, weight_init_constant=1.0):
        super().__init__()
        linear = partial(
            scaled_variance_linear, bias=False, init_constant=weight_init_constant
        )
        self.w1 = linear(in_features, hidden_features)
        self.w3 = linear(in_features, hidden_features)
        self.w2 = linear(hidden_features, in_features)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TimestepEmbedder(nn.Module):
    """Embeds a scalar timestep (or scalar conditioning) into a vector."""

    def __init__(self, hidden_size, frequency_embedding_size=256, init_constant=1.0):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        linear = partial(scaled_variance_linear, bias=True, init_constant=init_constant)
        self.mlp = nn.Sequential(
            linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings. t: (B,)."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


class LabelEmbedder(nn.Module):
    """Embeds class labels (index num_classes = null/unconditional label)."""

    def __init__(self, num_classes, hidden_size, init_constant=1.0):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        nn.init.normal_(
            self.embedding_table.weight, std=init_constant / math.sqrt(hidden_size)
        )

    def forward(self, labels):
        return self.embedding_table(labels)


class VideoPatchEmbedder(nn.Module):
    """Video latents to patch embedding via Conv3d with patch (1, 2, 2)."""

    def __init__(self, in_channels, hidden_size, patch_size=(1, 2, 2), bias=True):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )
        nn.init.xavier_uniform_(self.proj.weight.view(hidden_size, -1))
        if bias:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        # x: (B, C, T, H, W) -> (B, hidden, T', H', W') -> (B, N, hidden)
        # Token order: t-major, then h, then w.
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


#################################################################################
#                   Modern Transformer Components with Vec Gates                 #
#################################################################################


def apply_rotary_pos_emb(x, rope_cos, rope_sin):
    """Rotate adjacent-pair channels of x by precomputed angles.

    Matches the JAX complex-view convention: adjacent channel pairs
    (x[..., 2i], x[..., 2i+1]) form (real, imag).

    Args:
        x: (batch, seq_len, num_heads, head_dim)
        rope_cos, rope_sin: (seq_len, head_dim // 2)
    """
    x_f32 = x.float()
    x_real = x_f32[..., 0::2]
    x_imag = x_f32[..., 1::2]
    cos = rope_cos[None, :, None, :]
    sin = rope_sin[None, :, None, :]
    out_real = x_real * cos - x_imag * sin
    out_imag = x_real * sin + x_imag * cos
    out = torch.stack([out_real, out_imag], dim=-1).flatten(-2)
    return out.to(x.dtype)


class RoPEAttention(nn.Module):
    """Multi-head self-attention with RoPE, QK RMS norm, pluggable kernel."""

    def __init__(self, hidden_size, num_heads, weight_init_constant=1.0,
                 attn_impl=sdpa_math_attention):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attn_impl = attn_impl

        linear = partial(
            scaled_variance_linear, bias=False, init_constant=weight_init_constant
        )
        self.q_proj = linear(hidden_size, hidden_size)
        self.k_proj = linear(hidden_size, hidden_size)
        self.v_proj = linear(hidden_size, hidden_size)
        self.out_proj = linear(hidden_size, hidden_size)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, rope_cos, rope_sin):
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rotary_pos_emb(q, rope_cos, rope_sin)
        k = apply_rotary_pos_emb(k, rope_cos, rope_sin)

        attn = self.attn_impl(q, k, v)
        attn = attn.reshape(batch, seq_len, self.hidden_size)

        return self.out_proj(attn)


class TransformerBlock(nn.Module):
    """Transformer block with zero-initialized vector gates on residuals."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=8 / 3,
                 weight_init_constant=1.0, attn_impl=sdpa_math_attention):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = RoPEAttention(
            hidden_size,
            num_heads=num_heads,
            weight_init_constant=weight_init_constant,
            attn_impl=attn_impl,
        )
        self.norm2 = RMSNorm(hidden_size)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUMlp(
            hidden_size, mlp_hidden_dim, weight_init_constant=weight_init_constant
        )

        self.attn_scale = nn.Parameter(torch.zeros(hidden_size))
        self.mlp_scale = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin) * self.attn_scale
        x = x + self.mlp(self.norm2(x)) * self.mlp_scale
        return x


class FinalLayer(nn.Module):
    """Final projection layer with RMSNorm and zero-init weights."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        patch_t, patch_h, patch_w = patch_size
        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(
            hidden_size, patch_t * patch_h * patch_w * out_channels, bias=True
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(self.norm(x))


#################################################################################
#                           Rotary Position Helpers                              #
#################################################################################


def precompute_axial_rope_3d(
    head_dim, grid_size, num_prefix_tokens, axis_dims=None, theta=10000.0
):
    """3D axial RoPE angles for video patch tokens plus prefix tokens.

    head_dim is split into (t, h, w) parts (default 16/24/24 for head_dim 64);
    each part gets standard 1D RoPE on its axis position.

    Prefix conditioning tokens use the identity rotation (cos=1, sin=0), i.e.
    they are position-free: they are distinguished by their learnable token
    content, not by position, so no rotary phase is applied to them.

    Returns:
        rope_cos, rope_sin: (num_prefix_tokens + T*Gh*Gw, head_dim // 2) float32.
    """
    if axis_dims is None:
        dim_t = 2 * ((head_dim // 4) // 2)  # even split, t gets ~1/4
        dim_h = 2 * (((head_dim - dim_t) // 2) // 2)
        dim_w = head_dim - dim_t - dim_h
        axis_dims = (dim_t, dim_h, dim_w)
    assert sum(axis_dims) == head_dim
    assert all(d % 2 == 0 for d in axis_dims)

    grid_t, grid_h, grid_w = grid_size
    pos_t, pos_h, pos_w = torch.meshgrid(
        torch.arange(grid_t, dtype=torch.float32),
        torch.arange(grid_h, dtype=torch.float32),
        torch.arange(grid_w, dtype=torch.float32),
        indexing="ij",
    )  # each (T, Gh, Gw), matching t-major token order of VideoPatchEmbedder

    angle_parts = []
    for positions, axis_dim in zip(
        (pos_t.flatten(), pos_h.flatten(), pos_w.flatten()), axis_dims
    ):
        freqs = 1.0 / (
            theta ** (torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim)
        )
        angle_parts.append(torch.outer(positions, freqs))
    angles = torch.cat(angle_parts, dim=-1)  # (N, head_dim // 2)

    # Identity rotation (angle 0) on prefix conditioning tokens.
    prefix_angles = torch.zeros(num_prefix_tokens, head_dim // 2)
    angles = torch.cat([prefix_angles, angles], dim=0)

    return torch.cos(angles), torch.sin(angles)


#################################################################################
#                improved MeanFlow DiT with In-context Conditioning              #
#################################################################################


class IMFDiTVideo(nn.Module):
    """improved MeanFlow DiT for video latents.

    A shared backbone processes the first (depth - aux_head_depth) layers.
    Two heads of equal depth (aux_head_depth) branch off afterwards.
    """

    def __init__(
        self,
        input_size=16,
        num_frames=3,
        patch_size=(1, 2, 2),
        in_channels=16,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=8 / 3,
        num_classes=1000,
        aux_head_depth=8,
        num_class_tokens=8,
        num_time_tokens=4,
        num_cfg_tokens=4,
        num_interval_tokens=2,
        token_init_constant=1.0,
        embedding_init_constant=1.0,
        weight_init_constant=0.32,
        attn_impl=sdpa_math_attention,
        eval_mode=False,
    ):
        super().__init__()
        self.head_dim = hidden_size // num_heads
        assert self.head_dim == 64, "head_dim must be 64 for the flash-attn JVP op"

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.eval_mode = eval_mode

        patch_t, patch_h, patch_w = patch_size
        self.grid_size = (
            num_frames // patch_t,
            input_size // patch_h,
            input_size // patch_w,
        )
        num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.x_embedder = VideoPatchEmbedder(
            in_channels, hidden_size, patch_size=patch_size, bias=True
        )

        embed_kwargs = dict(
            hidden_size=hidden_size, init_constant=embedding_init_constant
        )
        self.h_embedder = TimestepEmbedder(**embed_kwargs)
        self.omega_embedder = TimestepEmbedder(**embed_kwargs)
        self.cfg_t_start_embedder = TimestepEmbedder(**embed_kwargs)
        self.cfg_t_end_embedder = TimestepEmbedder(**embed_kwargs)
        self.y_embedder = LabelEmbedder(num_classes, **embed_kwargs)

        token_std = token_init_constant / math.sqrt(hidden_size)
        self.time_tokens = nn.Parameter(
            torch.randn(num_time_tokens, hidden_size) * token_std
        )
        self.class_tokens = nn.Parameter(
            torch.randn(num_class_tokens, hidden_size) * token_std
        )
        self.omega_tokens = nn.Parameter(
            torch.randn(num_cfg_tokens, hidden_size) * token_std
        )
        self.t_min_tokens = nn.Parameter(
            torch.randn(num_interval_tokens, hidden_size) * token_std
        )
        self.t_max_tokens = nn.Parameter(
            torch.randn(num_interval_tokens, hidden_size) * token_std
        )

        self.prefix_tokens = (
            num_class_tokens
            + num_cfg_tokens
            + 2 * num_interval_tokens
            + num_time_tokens
        )

        rope_cos, rope_sin = precompute_axial_rope_3d(
            self.head_dim, self.grid_size, self.prefix_tokens
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        head_depth = aux_head_depth
        shared_depth = depth - head_depth

        block_kwargs = dict(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            weight_init_constant=weight_init_constant,
            attn_impl=attn_impl,
        )
        self.shared_blocks = nn.ModuleList(
            [TransformerBlock(**block_kwargs) for _ in range(shared_depth)]
        )
        self.u_heads = nn.ModuleList(
            [TransformerBlock(**block_kwargs) for _ in range(head_depth)]
        )
        # We don't need the v heads during evaluation
        self.v_heads = nn.ModuleList(
            [
                TransformerBlock(**block_kwargs)
                for _ in range(head_depth if not eval_mode else 0)
            ]
        )

        self.u_final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.v_final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

    def unpatchify(self, x):
        """(B, N, patch_t*patch_h*patch_w*C) -> (B, C, T, H, W)."""
        batch = x.shape[0]
        channels = self.out_channels
        patch_t, patch_h, patch_w = self.patch_size
        grid_t, grid_h, grid_w = self.grid_size
        assert grid_t * grid_h * grid_w == x.shape[1]

        x = x.reshape(
            batch, grid_t, grid_h, grid_w, patch_t, patch_h, patch_w, channels
        )
        x = torch.einsum("nthwpqrc->nctphqwr", x)
        return x.reshape(
            batch, channels, grid_t * patch_t, grid_h * patch_h, grid_w * patch_w
        )

    def _build_sequence(self, x, h, w, t_min, t_max, y):
        """
        Build the input token sequence for the transformer.
        1. Embed the input video latent patches.
        2. Embed the conditioning information (time, omega, cfg, class labels).
        3. Prepend the conditioning tokens to the patch embeddings.

        Args:
            x: Input video latents (B, C, T, H, W)
            h: timestep difference t - r, shape (B,)
            w: CFG scale, shape (B,)
            t_min, t_max: CFG interval, shape (B,)
            y: Class labels, shape (B,)
        """
        x_embed = self.x_embedder(x)
        h_embed = self.h_embedder(h)
        omega_embed = self.omega_embedder(1 - 1 / w)
        t_min_embed = self.cfg_t_start_embedder(t_min)
        t_max_embed = self.cfg_t_end_embedder(t_max)
        y_embed = self.y_embedder(y)

        time_tokens = self.time_tokens + h_embed.unsqueeze(1)
        omega_tokens = self.omega_tokens + omega_embed.unsqueeze(1)
        t_min_tokens = self.t_min_tokens + t_min_embed.unsqueeze(1)
        t_max_tokens = self.t_max_tokens + t_max_embed.unsqueeze(1)
        class_tokens = self.class_tokens + y_embed.unsqueeze(1)

        return torch.cat(
            [
                class_tokens,
                omega_tokens,
                t_min_tokens,
                t_max_tokens,
                time_tokens,
                x_embed,
            ],
            dim=1,
        )

    def forward(self, x, t, h, w, t_min, t_max, y):
        """
        Forward pass of the video imfDiT model.

        Args:
            x: Input video latents (B, C, T, H, W)
            t, h: time step and time difference t - r, shape (B,)
            w: CFG scale, shape (B,)
            t_min, t_max: CFG interval, shape (B,)
            y: Class labels, shape (B,)

        Returns:
            u: Average velocity field (B, C, T, H, W)
            v: Instantaneous velocity field (B, C, T, H, W)
        """
        # We don't explicitly condition on time t, only on h = t - r
        # following https://arxiv.org/abs/2502.13129
        seq = self._build_sequence(x, h, w, t_min, t_max, y)

        for block in self.shared_blocks:
            seq = block(seq, self.rope_cos, self.rope_sin)

        u_seq = v_seq = seq
        for block in self.u_heads:
            u_seq = block(u_seq, self.rope_cos, self.rope_sin)

        for block in self.v_heads:
            v_seq = block(v_seq, self.rope_cos, self.rope_sin)

        u_tokens = u_seq[:, self.prefix_tokens :]
        v_tokens = v_seq[:, self.prefix_tokens :]

        u = self.unpatchify(self.u_final_layer(u_tokens))
        v = self.unpatchify(self.v_final_layer(v_tokens))

        return u, v


#################################################################################
#                             iMF Video DiT Configs                              #
#################################################################################


def imf_dit_video_S(num_classes, **kwargs):
    return IMFDiTVideo(
        hidden_size=384,
        depth=8,
        num_heads=6,
        aux_head_depth=4,
        num_classes=num_classes,
        **kwargs,
    )


def imf_dit_video_B(num_classes, **kwargs):
    return IMFDiTVideo(
        hidden_size=768,
        depth=12,
        num_heads=12,
        aux_head_depth=8,
        num_classes=num_classes,
        **kwargs,
    )
