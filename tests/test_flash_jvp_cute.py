"""Test fused FlashAttention forward + JVP kernel against torch.func.jvp reference.

Reference: torch.func.jvp over F.scaled_dot_product_attention in fp32 math.
Pass threshold: max abs err <= 2e-2 and relative Frobenius err <= 2e-2 (bf16 inputs).
"""

import os
import sys

try:
    import pytest
except ImportError:  # allow running as a plain script without pytest installed
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.flash_jvp_kernel import flash_attn_jvp_func

SHAPES = [
    (2, 768, 6, 64),
    (1, 212, 6, 64),  # seqlen not a multiple of the tile size
    (2, 1024, 8, 64),
]


def reference_jvp(q, k, v, tq, tk, tv, softmax_scale):
    """torch.func.jvp of SDPA in fp32 math. Inputs (B, S, H, D)."""

    def attn(q_, k_, v_):
        # SDPA expects (B, H, S, D)
        return F.scaled_dot_product_attention(
            q_.transpose(1, 2), k_.transpose(1, 2), v_.transpose(1, 2),
            scale=softmax_scale,
        ).transpose(1, 2)

    primals = tuple(t.float() for t in (q, k, v))
    tangents = tuple(t.float() for t in (tq, tk, tv))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        o_ref, t_o_ref = torch.func.jvp(attn, primals, tangents)
    return o_ref, t_o_ref


def check(name, out, ref, max_abs_tol=2e-2, rel_fro_tol=2e-2):
    out = out.float()
    ref = ref.float()
    max_abs = (out - ref).abs().max().item()
    rel_fro = ((out - ref).norm() / ref.norm()).item()
    print(f"  {name}: max_abs_err={max_abs:.3e}  rel_fro_err={rel_fro:.3e}")
    assert max_abs <= max_abs_tol, f"{name}: max abs err {max_abs} > {max_abs_tol}"
    assert rel_fro <= rel_fro_tol, f"{name}: rel Frobenius err {rel_fro} > {rel_fro_tol}"


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_flash_jvp(shape, dtype):
    torch.manual_seed(0)
    B, S, H, D = shape
    device = "cuda"
    q, k, v, tq, tk, tv = [
        torch.randn(B, S, H, D, device=device, dtype=dtype) for _ in range(6)
    ]
    softmax_scale = D ** -0.5

    o, t_o, lse = flash_attn_jvp_func(q, k, v, tq, tk, tv, softmax_scale=softmax_scale)
    o_ref, t_o_ref = reference_jvp(q, k, v, tq, tk, tv, softmax_scale)

    print(f"\nshape={shape} dtype={dtype}")
    check("o", o, o_ref)
    check("t_o", t_o, t_o_ref)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_flash_jvp_lse(shape, dtype):
    """LSE output vs FA4 stock forward's lse on identical inputs."""
    from flash_attn.cute.interface import _flash_attn_fwd

    torch.manual_seed(0)
    B, S, H, D = shape
    device = "cuda"
    q, k, v, tq, tk, tv = [
        torch.randn(B, S, H, D, device=device, dtype=dtype) for _ in range(6)
    ]

    _, _, lse = flash_attn_jvp_func(q, k, v, tq, tk, tv)
    _, lse_ref, _, _ = _flash_attn_fwd(q, k, v, return_lse=True)

    assert lse.shape == lse_ref.shape == (B, H, S), (lse.shape, lse_ref.shape)
    max_abs = (lse - lse_ref).abs().max().item()
    print(f"\nshape={shape} lse max_abs_err={max_abs:.3e}")
    assert max_abs <= 1e-3, f"lse: max abs err {max_abs} > 1e-3"


def test_flash_jvp_prealloc_buffers():
    """Provided out/t_out/lse_out buffers are filled in place."""
    torch.manual_seed(1)
    B, S, H, D = 1, 212, 6, 64
    device = "cuda"
    q, k, v, tq, tk, tv = [
        torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16) for _ in range(6)
    ]
    out = torch.empty_like(q)
    t_out = torch.empty_like(q)
    lse_out = torch.empty(B, H, S, dtype=torch.float32, device=device)

    o, t_o, lse = flash_attn_jvp_func(
        q, k, v, tq, tk, tv, out=out, t_out=t_out, lse_out=lse_out
    )
    assert o is out and t_o is t_out and lse is lse_out

    o_ref, t_o_ref, lse_ref = flash_attn_jvp_func(q, k, v, tq, tk, tv)
    assert torch.equal(out, o_ref)
    assert torch.equal(t_out, t_o_ref)
    assert torch.equal(lse_out, lse_ref)
    print("\nprealloc buffers: in-place fill OK")


if __name__ == "__main__":
    for shape in SHAPES:
        test_flash_jvp(shape, torch.bfloat16)
        test_flash_jvp_lse(shape, torch.bfloat16)
    test_flash_jvp_prealloc_buffers()
    print("All tests passed.")
