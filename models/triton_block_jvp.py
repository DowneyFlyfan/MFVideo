"""Triton forward-mode JVP (Jacobian-vector product) of a full pre-LN
(pre-LayerNorm) transformer block, MeanFlow-style: the tangent is taken
w.r.t. the input x only, all parameter tangents are zero.

Block: x -> RMSNorm1 -> QKV linear -> multi-head attention -> out linear
         -> +residual -> RMSNorm2 -> SwiGLU MLP -> +residual

Because parameter tangents are zero, every linear layer applies the SAME
weight to primal and tangent, so each linear runs as ONE GEMM (general
matrix multiplication) on the stacked batch [x; dx] of shape (2B, S, D).

Speed path (sm_89+, torch._scaled_mm available):
  * All four linears run in fp8 (float8_e4m3fn) tensor cores via
    torch._scaled_mm with per-tensor scales: weights are amax-scaled to
    the fp8 range once and cached; activations are quantized to fp8
    *inside the producing Triton kernel* (RMSNorm JVP, flash JVP,
    SwiGLU JVP) with unit scale, so quantization costs no extra pass.
  * The flash-attention JVP kernel takes fp16 q/k/v (the QKV GEMM emits
    fp16 directly) and issues all tl.dot ops with fp16 accumulation
    (2x MMA rate on GeForce), accumulating across key tiles in fp32 so
    per-element error stays at the fp16 per-tile level.  The two
    tangent-score dots dQ K^T + Q dK^T are merged into a single dot of
    inner width 2*head_dim via tl.join interleaving.

All softmax statistics and cross-tile accumulation are fp32.

fp32 io mode (dispatch on input dtype): x/dx/params in fp32, outputs fp32.
Three variants selected by the `variant` argument:
  * "fast" (default): identical internals to the bf16 path — fp8 linears
    and fp16-accumulate attention.  The Triton kernels read fp32 and
    write fp8/fp16 directly (casts fused, no separate cast passes); the
    out/gate-up/down `_scaled_mm` calls emit fp32, so the residual
    stream and both outputs are fp32 end to end.
  * "hp" (high precision): fp16 weights + fp16 activations, cuBLAS fp16
    GEMMs (fp32 accumulation), the same fp16-accumulate flash JVP
    kernel, and an fp32 residual stream.  No fp8 anywhere, so error is
    fp16-rounding-limited (~1e-4 rel Fro) at near-fast-path speed.
  * "tf32": every `tl.dot` runs on fp32 operands with tf32 inner
    precision, the four linears run through cuBLAS with tf32 enabled
    (enabled locally inside this call only, global setting restored),
    and every intermediate buffer is fp32 — fp32-storage fallback
    (slower than "hp", similar accuracy).
"""

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl

__all__ = ["triton_block_jvp", "TritonBlockJVP", "ffn_dim", "init_block_params"]

_FP8 = torch.float8_e4m3fn
_FP8_MAX = 448.0
_ONES = {}


def _one(dev):
    t = _ONES.get(dev)
    if t is None:
        t = torch.ones(1, device=dev, dtype=torch.float32)
        _ONES[dev] = t
    return t


def _fp8_weight(w):
    """Quantize a weight matrix to fp8 with a per-tensor amax scale.
    Returns (w8, descale) with w ~= w8 * descale."""
    amax = w.detach().abs().amax().float().clamp_min(1e-12)
    descale = (amax / _FP8_MAX).reshape(1)
    w8 = (w.detach().float() / descale).clamp_(-_FP8_MAX, _FP8_MAX).to(_FP8)
    return w8, descale


def _fp8_mm(a8, w8_descale, out_dtype):
    """(M, K) fp8 row-major @ cached fp8 weight (N, K) -> (M, N)."""
    w8, descale = w8_descale
    return torch._scaled_mm(
        a8, w8.t(), scale_a=_one(a8.device), scale_b=descale,
        out_dtype=out_dtype,
    )


def ffn_dim(d_model: int) -> int:
    """SwiGLU hidden width: 8*d/3 rounded to a multiple of 64."""
    return max(64, int(round(8.0 * d_model / 3.0 / 64.0)) * 64)


# --------------------------------------------------------------------------
# RMSNorm JVP:  r = sqrt(mean(x^2)+eps);  y = w*x/r
#               dy = w*(dx/r - x*mean(x*dx)/r^3)
# Outputs are stored in fp8 (unit scale) as input to the following GEMM.
# --------------------------------------------------------------------------
@triton.jit
def _rmsnorm_jvp_kernel(X, DX, W, Y, DY, D, eps, BLOCK: tl.constexpr,
                        CLAMP: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(X + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    dx = tl.load(DX + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    inv_r = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)
    mxdx = tl.sum(x * dx, axis=0) / D
    y = w * x * inv_r
    dy = w * (dx * inv_r - x * (mxdx * inv_r * inv_r * inv_r))
    if CLAMP:
        y = tl.clamp(y, -448.0, 448.0)
        dy = tl.clamp(dy, -448.0, 448.0)
    tl.store(Y + row * D + cols, y.to(Y.dtype.element_ty), mask=mask)
    tl.store(DY + row * D + cols, dy.to(DY.dtype.element_ty), mask=mask)


def _rmsnorm_jvp(x, dx, w, y, dy, eps):
    n_rows, d = x.reshape(-1, x.shape[-1]).shape
    BLOCK = triton.next_power_of_2(d)
    num_warps = 4 if BLOCK <= 1024 else 8
    _rmsnorm_jvp_kernel[(n_rows,)](
        x, dx, w, y, dy, d, eps, BLOCK=BLOCK, num_warps=num_warps,
        CLAMP=(y.dtype == _FP8),
    )


# --------------------------------------------------------------------------
# Flash-attention JVP.  Per (m, n) tile, with softmax scale c:
#   S  = c * Q K^T                    (kept in log2 units for exp2)
#   dS = c * (dQ K^T + Q dK^T)        (one dot over width 2*hd, joined)
#   Ptil = exp(S - m)   (running max m; the m-shift cancels exactly in JVP)
#   T  = Ptil * dS
#   acc_o += Ptil @ V;  acc_do += T @ V + Ptil @ dV
#   l += rowsum(Ptil);  mu += rowsum(T);  rescale all on max update
#   o = acc_o/l;  do = acc_do/l - (mu/l)*o
# q/k/v are fp16; every tl.dot uses fp16 accumulation (per tile), the
# cross-tile accumulators are fp32.  o/do are stored in fp8.
# --------------------------------------------------------------------------
_FLASH_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
]

# fp32 operands double the k/dk/v/dv tile footprint; only these configs
# fit the 99 KB shared-memory limit of sm_120 (measured via warmup).
_FLASH_CONFIGS_FP32 = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=3),
]


@triton.jit
def _flash_jvp_kernel_impl(
    Q, K, V, DQ, DK, DV, O, DO,
    s_qb, s_qs, s_qh,          # strides of the (B, S, H, hd) qkv views
    s_ob, s_os, s_oh,          # strides of the (B, S, H, hd) output views
    H, S, scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_S: tl.constexpr,
    FP16_MMA: tl.constexpr,    # fp16-accumulate dots (fp16 io) vs tf32 dots
    CLAMP: tl.constexpr,       # clamp outputs into the fp8 range
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    base = b * s_qb + h * s_qh
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < S

    qp = base + offs_m[:, None] * s_qs + offs_d[None, :]
    if EVEN_S:
        q = tl.load(Q + qp)
        dq = tl.load(DQ + qp)
    else:
        q = tl.load(Q + qp, mask=mask_m[:, None], other=0.0)
        dq = tl.load(DQ + qp, mask=mask_m[:, None], other=0.0)
    # interleaved [dq|q] so one dot of width 2*hd gives dQ K^T + Q dK^T
    qj = tl.reshape(tl.join(dq, q), (BLOCK_M, 2 * HEAD_DIM))

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    mu_i = tl.zeros([BLOCK_M], tl.float32)
    acc_o = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    acc_do = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    LOG2E: tl.constexpr = 1.4426950408889634
    qk_scale = scale * LOG2E

    for n0 in range(0, S, BLOCK_N):
        offs = n0 + offs_n
        kp = base + offs[:, None] * s_qs + offs_d[None, :]
        if EVEN_S:
            k = tl.load(K + kp)
            dk = tl.load(DK + kp)
        else:
            mask_n = offs < S
            k = tl.load(K + kp, mask=mask_n[:, None], other=0.0)
            dk = tl.load(DK + kp, mask=mask_n[:, None], other=0.0)
        kj = tl.reshape(tl.join(k, dk), (BLOCK_N, 2 * HEAD_DIM))

        if FP16_MMA:
            s_f = tl.dot(q, tl.trans(k), out_dtype=tl.float16).to(tl.float32)
            ds = tl.dot(qj, tl.trans(kj),
                        out_dtype=tl.float16).to(tl.float32) * scale
        else:
            s_f = tl.dot(q, tl.trans(k))
            ds = tl.dot(qj, tl.trans(kj)) * scale
        if EVEN_S:
            s2 = s_f * qk_scale
        else:
            s2 = tl.where(mask_n[None, :], s_f * qk_scale, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s2, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s2 - m_new[:, None])
        t = p * ds

        l_i = l_i * alpha + tl.sum(p, 1)
        mu_i = mu_i * alpha + tl.sum(t, 1)

        if EVEN_S:
            v = tl.load(V + kp)
            dv = tl.load(DV + kp)
        else:
            v = tl.load(V + kp, mask=mask_n[:, None], other=0.0)
            dv = tl.load(DV + kp, mask=mask_n[:, None], other=0.0)
        if FP16_MMA:
            pc = p.to(tl.float16)
            tc = t.to(tl.float16)
            o_t = tl.dot(pc, v, out_dtype=tl.float16).to(tl.float32)
            do_t = (tl.dot(tc, v, out_dtype=tl.float16)
                    + tl.dot(pc, dv, out_dtype=tl.float16)).to(tl.float32)
        else:
            o_t = tl.dot(p, v)
            do_t = tl.dot(t, v) + tl.dot(p, dv)
        acc_o = acc_o * alpha[:, None] + o_t
        acc_do = acc_do * alpha[:, None] + do_t
        m_i = m_new

    o = acc_o / l_i[:, None]
    do = acc_do / l_i[:, None] - (mu_i / l_i)[:, None] * o
    if CLAMP:
        o = tl.clamp(o, -448.0, 448.0)
        do = tl.clamp(do, -448.0, 448.0)

    op = b * s_ob + h * s_oh + offs_m[:, None] * s_os + offs_d[None, :]
    if EVEN_S:
        tl.store(O + op, o.to(O.dtype.element_ty))
        tl.store(DO + op, do.to(DO.dtype.element_ty))
    else:
        tl.store(O + op, o.to(O.dtype.element_ty), mask=mask_m[:, None])
        tl.store(DO + op, do.to(DO.dtype.element_ty), mask=mask_m[:, None])


# Two autotuner entry points over the same kernel body: fp16 operands
# and fp32 (tf32-dot) operands need disjoint tile-config spaces.
_flash_jvp_kernel = triton.autotune(
    configs=_FLASH_CONFIGS, key=["S", "H"])(_flash_jvp_kernel_impl)
_flash_jvp_kernel_fp32 = triton.autotune(
    configs=_FLASH_CONFIGS_FP32, key=["S", "H"])(_flash_jvp_kernel_impl)


def _flash_jvp(q, k, v, dq, dk, dv, o, do, scale):
    B, S, H, hd = q.shape
    assert q.stride() == k.stride() == v.stride() == dq.stride()
    assert q.stride(-1) == 1 and o.stride(-1) == 1
    fp16 = q.dtype == torch.float16
    kernel = _flash_jvp_kernel if fp16 else _flash_jvp_kernel_fp32
    grid = lambda meta: (triton.cdiv(S, meta["BLOCK_M"]), B * H)
    kernel[grid](
        q, k, v, dq, dk, dv, o, do,
        q.stride(0), q.stride(1), q.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        H, S, scale, HEAD_DIM=hd,
        # 128 is the largest BLOCK_M/BLOCK_N in either autotune space, so
        # divisibility by 128 implies no bounds masks for any config.
        EVEN_S=(S % 128 == 0),
        FP16_MMA=fp16,
        CLAMP=(o.dtype == _FP8),
    )


# --------------------------------------------------------------------------
# SwiGLU JVP:  h = silu(g)*u
#              dh = silu'(g)*dg*u + silu(g)*du
#              silu'(g) = sigmoid(g)*(1 + g*(1 - sigmoid(g)))
# GU rows are [gate | up] of width 2F; outputs (fp8) are width F.
# --------------------------------------------------------------------------
@triton.jit
def _swiglu_jvp_kernel(GU, DGU, Hout, DHout, F, total, BLOCK: tl.constexpr,
                       CLAMP: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < total
    row = i // F
    col = i - row * F
    base = row * (2 * F) + col
    g = tl.load(GU + base, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(GU + base + F, mask=mask, other=0.0).to(tl.float32)
    dg = tl.load(DGU + base, mask=mask, other=0.0).to(tl.float32)
    du = tl.load(DGU + base + F, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g)
    silu = g * sig
    dsilu = sig * (1.0 + g * (1.0 - sig))
    h = silu * u
    dh = dsilu * dg * u + silu * du
    if CLAMP:
        h = tl.clamp(h, -448.0, 448.0)
        dh = tl.clamp(dh, -448.0, 448.0)
    tl.store(Hout + i, h.to(Hout.dtype.element_ty), mask=mask)
    tl.store(DHout + i, dh.to(DHout.dtype.element_ty), mask=mask)


def _swiglu_jvp(gu, dgu, h, dh):
    f = gu.shape[-1] // 2
    total = h.numel()
    BLOCK = 1024
    _swiglu_jvp_kernel[(triton.cdiv(total, BLOCK),)](
        gu, dgu, h, dh, f, total, BLOCK=BLOCK, num_warps=4,
        CLAMP=(h.dtype == _FP8),
    )


# --------------------------------------------------------------------------
# Full block JVP
# --------------------------------------------------------------------------
def _fp8_weight_cache(params, w_gu):
    return {
        "w_qkv": _fp8_weight(params["w_qkv"]),
        "w_out": _fp8_weight(params["w_out"]),
        "w_gate_up": _fp8_weight(w_gu),
        "w_down": _fp8_weight(params["w_down"]),
    }


def _fp16_weight_cache(params, w_gu):
    return {
        "w_qkv": params["w_qkv"].detach().half(),
        "w_out": params["w_out"].detach().half(),
        "w_gate_up": w_gu.detach().half(),
        "w_down": params["w_down"].detach().half(),
    }


def _block_jvp_hp(x, dx, params, eps):
    """fp32-io high-precision variant: fp16 weights + fp16 activations,
    cuBLAS fp16 GEMMs, the fp16-accumulate flash JVP kernel, and an fp32
    residual stream (residual adds and both outputs are fp32).  Error is
    fp16-rounding-limited (~5e-4 rel Fro), no fp8 anywhere."""
    B, S, D = x.shape
    n_heads = params["n_heads"]
    hd = D // n_heads
    assert hd * n_heads == D
    scale = 1.0 / math.sqrt(hd)
    dev = x.device

    w_gu = params.get("w_gate_up")
    if w_gu is None:
        w_gu = torch.cat([params["w_gate"], params["w_up"]], dim=0)
    f = w_gu.shape[0] // 2
    w16 = params.get("_fp16")
    if w16 is None:
        w16 = _fp16_weight_cache(params, w_gu)
    M = 2 * B * S

    # ---- RMSNorm1: fp32 in, fp16 out (cast fused into the kernel)
    xs16 = torch.empty(2 * B, S, D, device=dev, dtype=torch.float16)
    _rmsnorm_jvp(x, dx, params["w_norm1"], xs16[:B], xs16[B:], eps)

    # ---- QKV projection: one fp16 GEMM on the stacked batch
    qkv = (xs16.view(M, D) @ w16["w_qkv"].t()).view(2 * B, S, 3, n_heads, hd)
    q, k, v = qkv[:B, :, 0], qkv[:B, :, 1], qkv[:B, :, 2]
    dq, dk, dv = qkv[B:, :, 0], qkv[B:, :, 1], qkv[B:, :, 2]

    # ---- fused flash-attention JVP (fp16 io, fp32 softmax statistics)
    attn16 = torch.empty(2 * B, S, D, device=dev, dtype=torch.float16)
    _flash_jvp(q, k, v, dq, dk, dv,
               attn16[:B].view(B, S, n_heads, hd),
               attn16[B:].view(B, S, n_heads, hd), scale)

    # ---- out projection + residual add into an fp32 residual stream
    proj = (attn16.view(M, D) @ w16["w_out"].t()).view(2 * B, S, D)
    res = torch.empty(2 * B, S, D, device=dev, dtype=torch.float32)
    torch.add(proj[:B], x, out=res[:B])
    torch.add(proj[B:], dx, out=res[B:])

    # ---- RMSNorm2 (fp32 in, fp16 out, reuse xs16)
    _rmsnorm_jvp(res[:B], res[B:], params["w_norm2"], xs16[:B], xs16[B:], eps)

    # ---- gate/up projection (fp16 GEMM, fused weight)
    gu = (xs16.view(M, D) @ w16["w_gate_up"].t()).view(2 * B, S, 2 * f)

    # ---- fused SwiGLU JVP (fp32 math inside, fp16 out)
    act16 = torch.empty(2 * B, S, f, device=dev, dtype=torch.float16)
    _swiglu_jvp(gu[:B], gu[B:], act16[:B], act16[B:])

    # ---- down projection + fp32 residual add
    res += (act16.view(M, f) @ w16["w_down"].t()).view(2 * B, S, D)
    return res[:B].contiguous(), res[B:].contiguous()


def _tf32_on():
    """Enable cuBLAS tf32, returning a restore closure (torch>=2.9 uses
    fp32_precision, allow_tf32 elsewhere)."""
    m = torch.backends.cuda.matmul
    if hasattr(m, "fp32_precision"):
        old = m.fp32_precision
        m.fp32_precision = "tf32"
        def restore():
            m.fp32_precision = old
    else:
        old = m.allow_tf32
        m.allow_tf32 = True
        def restore():
            m.allow_tf32 = old
    return restore


def _block_jvp_tf32(x, dx, params, eps):
    """fp32-io variant: cuBLAS tf32 GEMMs, tf32 tl.dot in the flash
    kernel, all intermediates fp32.  Structure mirrors the fast path."""
    B, S, D = x.shape
    n_heads = params["n_heads"]
    hd = D // n_heads
    assert hd * n_heads == D
    scale = 1.0 / math.sqrt(hd)
    dev = x.device

    w_gu = params.get("w_gate_up")
    if w_gu is None:
        w_gu = torch.cat([params["w_gate"], params["w_up"]], dim=0)
    f = w_gu.shape[0] // 2
    M = 2 * B * S

    restore = _tf32_on()
    try:
        xs = torch.empty(2 * B, S, D, device=dev, dtype=torch.float32)
        _rmsnorm_jvp(x, dx, params["w_norm1"], xs[:B], xs[B:], eps)

        qkv = (xs.view(M, D) @ params["w_qkv"].t())
        qkv = qkv.view(2 * B, S, 3, n_heads, hd)
        q, k, v = qkv[:B, :, 0], qkv[:B, :, 1], qkv[:B, :, 2]
        dq, dk, dv = qkv[B:, :, 0], qkv[B:, :, 1], qkv[B:, :, 2]

        attn = torch.empty(2 * B, S, D, device=dev, dtype=torch.float32)
        _flash_jvp(q, k, v, dq, dk, dv,
                   attn[:B].view(B, S, n_heads, hd),
                   attn[B:].view(B, S, n_heads, hd), scale)

        res = (attn.view(M, D) @ params["w_out"].t()).view(2 * B, S, D)
        res[:B] += x
        res[B:] += dx

        _rmsnorm_jvp(res[:B], res[B:], params["w_norm2"], xs[:B], xs[B:], eps)

        gu = (xs.view(M, D) @ w_gu.t()).view(2 * B, S, 2 * f)

        act = torch.empty(2 * B, S, f, device=dev, dtype=torch.float32)
        _swiglu_jvp(gu[:B], gu[B:], act[:B], act[B:])

        out = (act.view(M, f) @ params["w_down"].t()).view(2 * B, S, D)
        out += res
    finally:
        restore()
    return out[:B].contiguous(), out[B:].contiguous()


def triton_block_jvp(x, dx, params, eps: float = 1e-6, variant: str = "fast"):
    """Forward-mode JVP of the transformer block w.r.t. the input only.

    x, dx : (B, S, D) bf16/fp16/fp32 CUDA tensors (contiguous).  Params
            must match the input dtype.  With fp32 inputs, `variant`
            selects the internals: "fast" (default; fp8 linears +
            fp16-accumulate attention, fp32 io), "hp" (fp16 weights and
            activations, fp32 residual stream, ~1e-4 rel error), or
            "tf32" (tf32 dots and cuBLAS tf32 GEMMs everywhere, fp32
            storage).  bf16/fp16 inputs always take the fast path;
            `variant` is ignored.
    params: dict with keys w_norm1 (D,), w_qkv (3D, D), w_out (D, D),
            w_norm2 (D,), w_gate (F, D), w_up (F, D), w_down (D, F),
            n_heads (int).  Optional key w_gate_up (2F, D) = cat(gate, up)
            avoids a per-call concatenation; optional key _fp8 (the dict
            built by _fp8_weight_cache) avoids per-call weight
            quantization.
    Returns (y, dy), each (B, S, D), in the input dtype.
    """
    assert x.is_cuda and x.is_contiguous() and dx.is_contiguous()
    if x.dtype == torch.float32 and variant == "tf32":
        return _block_jvp_tf32(x, dx, params, eps)
    if x.dtype == torch.float32 and variant == "hp":
        return _block_jvp_hp(x, dx, params, eps)
    B, S, D = x.shape
    n_heads = params["n_heads"]
    hd = D // n_heads
    assert hd * n_heads == D
    scale = 1.0 / math.sqrt(hd)
    dev, dt = x.device, x.dtype

    w_gu = params.get("w_gate_up")
    if w_gu is None:
        w_gu = torch.cat([params["w_gate"], params["w_up"]], dim=0)
    f = w_gu.shape[0] // 2

    fp8w = params.get("_fp8")
    if fp8w is None:
        fp8w = _fp8_weight_cache(params, w_gu)

    M = 2 * B * S

    # ---- RMSNorm1: write primal/tangent (fp8) into the stacked buffer
    xs8 = torch.empty(2 * B, S, D, device=dev, dtype=_FP8)
    _rmsnorm_jvp(x, dx, params["w_norm1"], xs8[:B], xs8[B:], eps)

    # ---- QKV projection: one fp8 GEMM on the stacked batch, fp16 out
    qkv = _fp8_mm(xs8.view(M, D), fp8w["w_qkv"], torch.float16)
    qkv = qkv.view(2 * B, S, 3, n_heads, hd)
    q, k, v = qkv[:B, :, 0], qkv[:B, :, 1], qkv[:B, :, 2]
    dq, dk, dv = qkv[B:, :, 0], qkv[B:, :, 1], qkv[B:, :, 2]

    # ---- fused flash-attention JVP, o/do written (fp8) into a buffer
    attn8 = torch.empty(2 * B, S, D, device=dev, dtype=_FP8)
    _flash_jvp(q, k, v, dq, dk, dv,
               attn8[:B].view(B, S, n_heads, hd),
               attn8[B:].view(B, S, n_heads, hd), scale)

    # ---- out projection (stacked fp8 GEMM) + residual add
    res = _fp8_mm(attn8.view(M, D), fp8w["w_out"], dt).view(2 * B, S, D)
    res[:B] += x
    res[B:] += dx

    # ---- RMSNorm2 (reuse xs8 as the stacked normed fp8 buffer)
    _rmsnorm_jvp(res[:B], res[B:], params["w_norm2"], xs8[:B], xs8[B:], eps)

    # ---- gate/up projection: one fp8 GEMM on stacked batch, fused weight
    gu = _fp8_mm(xs8.view(M, D), fp8w["w_gate_up"], dt).view(2 * B, S, 2 * f)

    # ---- fused SwiGLU JVP into a stacked fp8 buffer
    act8 = torch.empty(2 * B, S, f, device=dev, dtype=_FP8)
    _swiglu_jvp(gu[:B], gu[B:], act8[:B], act8[B:])

    # ---- down projection (stacked fp8 GEMM) + residual add
    out = _fp8_mm(act8.view(M, f), fp8w["w_down"], dt).view(2 * B, S, D)
    out += res
    return out[:B].contiguous(), out[B:].contiguous()


def init_block_params(d_model, n_heads, device="cuda", dtype=torch.bfloat16,
                      seed=0):
    """Reference init: N(0, 0.02) linears, 1 + 0.1*N(0,1) norm gains."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    f = ffn_dim(d_model)

    def lin(o, i):
        return (torch.randn(o, i, generator=g) * 0.02)

    def gain(n):
        return 1.0 + 0.1 * torch.randn(n, generator=g)

    p = {
        "w_norm1": gain(d_model),
        "w_qkv": lin(3 * d_model, d_model),
        "w_out": lin(d_model, d_model),
        "w_norm2": gain(d_model),
        "w_gate": lin(f, d_model),
        "w_up": lin(f, d_model),
        "w_down": lin(d_model, f),
    }
    p = {k: v.to(device=device, dtype=dtype) for k, v in p.items()}
    p["n_heads"] = n_heads
    return p


class TritonBlockJVP(nn.Module):
    """Module wrapper holding the block parameters (same init as the
    PyTorch reference).  forward(x, dx) -> (y, dy).  Caches the fused
    gate/up weight and the fp8/fp16-quantized weights across calls.
    With fp32 parameters/inputs, `variant` picks "fast" (fp8/fp16
    internals), "hp" (fp16 weights/activations, fp32 residual stream),
    or "tf32" (tf32 everywhere)."""

    def __init__(self, d_model, n_heads, device="cuda",
                 dtype=torch.bfloat16, eps=1e-6, seed=0, variant="fast"):
        super().__init__()
        self.n_heads = n_heads
        self.eps = eps
        self.variant = variant
        p = init_block_params(d_model, n_heads, device, dtype, seed)
        for name in ("w_norm1", "w_qkv", "w_out", "w_norm2",
                     "w_gate", "w_up", "w_down"):
            setattr(self, name, nn.Parameter(p[name]))
        self._w_gate_up = None
        self._fp8 = None
        self._fp16 = None

    def load_params(self, params):
        with torch.no_grad():
            for name in ("w_norm1", "w_qkv", "w_out", "w_norm2",
                         "w_gate", "w_up", "w_down"):
                getattr(self, name).copy_(params[name])
        self._w_gate_up = None
        self._fp8 = None
        self._fp16 = None

    def forward(self, x, dx):
        if self._w_gate_up is None or self._w_gate_up.dtype != x.dtype:
            self._w_gate_up = torch.cat(
                [self.w_gate.detach(), self.w_up.detach()], dim=0)
            self._fp8 = None
            self._fp16 = None
        fp32_variant = x.dtype == torch.float32 and self.variant in (
            "tf32", "hp")
        need_fp8 = not fp32_variant
        if need_fp8 and self._fp8 is None:
            self._fp8 = _fp8_weight_cache(
                {"w_qkv": self.w_qkv, "w_out": self.w_out,
                 "w_down": self.w_down}, self._w_gate_up)
        need_fp16 = x.dtype == torch.float32 and self.variant == "hp"
        if need_fp16 and self._fp16 is None:
            self._fp16 = _fp16_weight_cache(
                {"w_qkv": self.w_qkv, "w_out": self.w_out,
                 "w_down": self.w_down}, self._w_gate_up)
        params = {
            "w_norm1": self.w_norm1, "w_qkv": self.w_qkv,
            "w_out": self.w_out, "w_norm2": self.w_norm2,
            "w_gate": self.w_gate, "w_up": self.w_up,
            "w_down": self.w_down, "w_gate_up": self._w_gate_up,
            "_fp8": self._fp8, "_fp16": self._fp16,
            "n_heads": self.n_heads,
        }
        return triton_block_jvp(x, dx, params, eps=self.eps,
                                variant=self.variant)
