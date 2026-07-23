"""Correctness + benchmark for the Triton transformer-block JVP
(Jacobian-vector product).

bf16 io: reference is the pure PyTorch pre-LN block (math SDPA forced)
under torch.func.jvp in fp32, on the SAME bf16-representable weights.
Pass threshold: relative Frobenius error <= 3e-2 on y and dy.

fp32 io: reference is the same block under torch.func.jvp in fp64
(falls back to fp32 with a printed note if the fp64 math-SDPA OOMs).
Thresholds: "fast" variant <= 3e-2, "tf32" variant <= 3e-3.

Run:  /opt/miniconda3/bin/python3 tests/test_triton_block_jvp.py
"""

import os
import sys

try:
    import pytest
except ImportError:  # allow running as a plain script without pytest
    class _FakePytest:
        class mark:
            @staticmethod
            def parametrize(*a, **kw):
                def deco(fn):
                    return fn
                return deco

    pytest = _FakePytest()

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.triton_block_jvp import (
    triton_block_jvp,
    TritonBlockJVP,
    init_block_params,
)

EPS = 1e-6

# (B, S, D, n_heads)
CORRECTNESS_SHAPES = [
    (2, 512, 768, 12),
    (1, 1000, 768, 12),   # seqlen not a multiple of the tile size
    (2, 4096, 512, 8),
]

BENCH_SHAPES = [
    (2, 1024, 768, 12),
    (2, 4096, 768, 12),
    (1, 8192, 768, 12),
]


# --------------------------------------------------------------------------
# PyTorch reference block (eager, math SDPA — the only jvp-capable path)
# --------------------------------------------------------------------------
def ref_block(x, p, n_heads, eps=EPS):
    def rms(t, w):
        return w * t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps)

    B, S, D = x.shape
    hd = D // n_heads
    h = rms(x, p["w_norm1"])
    qkv = (h @ p["w_qkv"].t()).view(B, S, 3, n_heads, hd)
    q = qkv[:, :, 0].transpose(1, 2)
    k = qkv[:, :, 1].transpose(1, 2)
    v = qkv[:, :, 2].transpose(1, 2)
    with sdpa_kernel([SDPBackend.MATH]):
        a = F.scaled_dot_product_attention(q, k, v)
    a = a.transpose(1, 2).reshape(B, S, D)
    x = x + a @ p["w_out"].t()
    h2 = rms(x, p["w_norm2"])
    x = x + (F.silu(h2 @ p["w_gate"].t()) * (h2 @ p["w_up"].t())) @ p["w_down"].t()
    return x


def ref_jvp(x, dx, params, n_heads):
    return torch.func.jvp(
        lambda t: ref_block(t, params, n_heads), (x,), (dx,)
    )


def ref_jvp_f64(x, dx, params, n_heads):
    """fp64 reference; falls back to fp32 (with a note) if fp64 OOMs."""
    p64 = {k: (v.double() if torch.is_tensor(v) else v)
           for k, v in params.items()}
    try:
        y, dy = ref_jvp(x.double(), dx.double(), p64, n_heads)
        return y, dy, "fp64"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        p32 = {k: (v.float() if torch.is_tensor(v) else v)
               for k, v in params.items()}
        y, dy = ref_jvp(x.float(), dx.float(), p32, n_heads)
        print("  [fp64 reference OOM -> using fp32 reference]")
        return y, dy, "fp32"


def _err(name, out, ref, rel_tol=3e-2):
    out, ref = out.float(), ref.float()
    max_abs = (out - ref).abs().max().item()
    rel_fro = ((out - ref).norm() / ref.norm()).item()
    print(f"  {name}: rel_fro_err={rel_fro:.3e}  max_abs_err={max_abs:.3e}")
    assert torch.isfinite(out).all(), f"{name}: non-finite values"
    assert rel_fro <= rel_tol, f"{name}: rel Frobenius {rel_fro} > {rel_tol}"
    return rel_fro, max_abs


# --------------------------------------------------------------------------
# Correctness
# --------------------------------------------------------------------------
@pytest.mark.parametrize("shape", CORRECTNESS_SHAPES)
def test_block_jvp(shape):
    torch.manual_seed(0)
    B, S, D, H = shape
    dev = "cuda"
    params = init_block_params(D, H, device=dev, dtype=torch.bfloat16, seed=0)

    x = torch.randn(B, S, D, device=dev, dtype=torch.bfloat16)
    dx = torch.randn(B, S, D, device=dev, dtype=torch.bfloat16)

    y, dy = triton_block_jvp(x, dx, params, eps=EPS)

    # fp32 reference on the same bf16-representable weights/inputs
    p32 = {k: (v.float() if torch.is_tensor(v) else v) for k, v in params.items()}
    y_ref, dy_ref = ref_jvp(x.float(), dx.float(), p32, H)

    print(f"\nshape (B,S,D,H)={shape}")
    _err("y", y, y_ref)
    _err("dy", dy, dy_ref)


@pytest.mark.parametrize("shape", CORRECTNESS_SHAPES)
@pytest.mark.parametrize("variant,tol", [("fast", 3e-2), ("tf32", 3e-3)])
def test_block_jvp_fp32(shape, variant, tol):
    torch.manual_seed(0)
    B, S, D, H = shape
    dev = "cuda"
    params = init_block_params(D, H, device=dev, dtype=torch.float32, seed=0)

    x = torch.randn(B, S, D, device=dev, dtype=torch.float32)
    dx = torch.randn(B, S, D, device=dev, dtype=torch.float32)

    y, dy = triton_block_jvp(x, dx, params, eps=EPS, variant=variant)
    assert y.dtype == torch.float32 and dy.dtype == torch.float32

    y_ref, dy_ref, ref_prec = ref_jvp_f64(x, dx, params, H)

    print(f"\nfp32 io, variant={variant}, ref={ref_prec}, "
          f"shape (B,S,D,H)={shape}")
    _err("y", y.double(), y_ref.double(), rel_tol=tol)
    _err("dy", dy.double(), dy_ref.double(), rel_tol=tol)
    del y_ref, dy_ref
    torch.cuda.empty_cache()


def test_module_matches_functional():
    torch.manual_seed(0)
    B, S, D, H = 2, 512, 768, 12
    dev = "cuda"
    params = init_block_params(D, H, device=dev, dtype=torch.bfloat16, seed=3)
    mod = TritonBlockJVP(D, H, device=dev, dtype=torch.bfloat16, seed=3)
    x = torch.randn(B, S, D, device=dev, dtype=torch.bfloat16)
    dx = torch.randn(B, S, D, device=dev, dtype=torch.bfloat16)
    y1, dy1 = triton_block_jvp(x, dx, params, eps=EPS)
    with torch.no_grad():
        y2, dy2 = mod(x, dx)
    assert torch.equal(y1, y2) and torch.equal(dy1, dy2)
    print("\nmodule == functional: bit-exact")


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------
def _time_cuda(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def benchmark():
    dev = "cuda"
    print("\n=== Benchmark: bf16, CUDA events, 30 iters / 5 warmup ===")
    rows = []
    for B, S, D, H in BENCH_SHAPES:
        torch.manual_seed(0)
        params = init_block_params(D, H, device=dev, dtype=torch.bfloat16)
        mod = TritonBlockJVP(D, H, device=dev, dtype=torch.bfloat16)
        mod.load_params(params)
        x = torch.randn(B, S, D, device=dev, dtype=torch.bfloat16)
        dx = torch.randn(B, S, D, device=dev, dtype=torch.bfloat16)

        with torch.no_grad():
            t_triton = _time_cuda(lambda: mod(x, dx))

        def baseline():
            with torch.no_grad():
                return ref_jvp(x, dx, params, H)

        try:
            t_ref = _time_cuda(baseline)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            t_ref = float("nan")

        try:
            compiled = torch.compile(
                lambda a, b: ref_jvp(a, b, params, H), dynamic=False
            )
            with torch.no_grad():
                t_comp = _time_cuda(lambda: compiled(x, dx))
            comp_note = f"{t_comp:8.3f}"
        except Exception as e:  # noqa: BLE001
            t_comp = float("nan")
            comp_note = f"FAILED ({type(e).__name__})"
            torch.cuda.empty_cache()

        speed = t_ref / t_triton
        speed_c = t_comp / t_triton
        rows.append((B, S, D, H, t_triton, t_ref, speed, comp_note, speed_c))
        print(
            f"(B={B}, S={S}, D={D}, H={H})  triton={t_triton:8.3f} ms  "
            f"torch.func.jvp={t_ref:8.3f} ms  speedup={speed:6.2f}x  "
            f"compiled={comp_note} ms  speedup_vs_compiled={speed_c:6.2f}x"
        )
        del params, mod
        torch.cuda.empty_cache()
    return rows


def benchmark_fp32():
    dev = "cuda"
    tf32_flag = torch.backends.cuda.matmul.allow_tf32
    fp32_prec = getattr(torch.backends.cuda.matmul, "fp32_precision", "n/a")
    print(f"\n=== Benchmark: fp32 io, CUDA events, 30 iters / 5 warmup ===")
    print(f"baseline torch defaults: matmul.allow_tf32={tf32_flag}, "
          f"matmul.fp32_precision={fp32_prec} (NOT changed for baselines)")
    for B, S, D, H in BENCH_SHAPES:
        torch.manual_seed(0)
        params = init_block_params(D, H, device=dev, dtype=torch.float32)
        mod_fast = TritonBlockJVP(D, H, device=dev, dtype=torch.float32,
                                  variant="fast")
        mod_fast.load_params(params)
        mod_tf32 = TritonBlockJVP(D, H, device=dev, dtype=torch.float32,
                                  variant="tf32")
        mod_tf32.load_params(params)
        x = torch.randn(B, S, D, device=dev, dtype=torch.float32)
        dx = torch.randn(B, S, D, device=dev, dtype=torch.float32)

        with torch.no_grad():
            t_fast = _time_cuda(lambda: mod_fast(x, dx))
            t_tf32 = _time_cuda(lambda: mod_tf32(x, dx))

        def baseline():
            with torch.no_grad():
                return ref_jvp(x, dx, params, H)

        try:
            t_ref = _time_cuda(baseline)
            ref_note = f"{t_ref:8.3f}"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            t_ref = float("nan")
            ref_note = "OOM"

        try:
            compiled = torch.compile(
                lambda a, b: ref_jvp(a, b, params, H), dynamic=False
            )
            with torch.no_grad():
                t_comp = _time_cuda(lambda: compiled(x, dx))
            comp_note = f"{t_comp:8.3f}"
        except Exception as e:  # noqa: BLE001
            t_comp = float("nan")
            comp_note = f"FAILED ({type(e).__name__})"
            torch.cuda.empty_cache()

        print(
            f"(B={B}, S={S}, D={D}, H={H})  "
            f"fast={t_fast:8.3f} ms  tf32={t_tf32:8.3f} ms  "
            f"eager={ref_note} ms  compiled={comp_note} ms\n"
            f"    fast: {t_ref / t_fast:6.2f}x vs eager, "
            f"{t_comp / t_fast:6.2f}x vs compiled | "
            f"tf32: {t_ref / t_tf32:6.2f}x vs eager, "
            f"{t_comp / t_tf32:6.2f}x vs compiled"
        )
        del params, mod_fast, mod_tf32
        torch.cuda.empty_cache()


if __name__ == "__main__":
    for shape in CORRECTNESS_SHAPES:
        test_block_jvp(shape)
    for shape in CORRECTNESS_SHAPES:
        for variant, tol in [("fast", 3e-2), ("tf32", 3e-3)]:
            test_block_jvp_fp32(shape, variant, tol)
    test_module_matches_functional()
    print("\nAll correctness tests passed.")
    benchmark()
    benchmark_fp32()
