"""Attention op combining FlashAttention-4 (primal forward / reverse backward)
with a custom CuTeDSL fused JVP kernel (forward-mode tangent).

Layout convention: (batch, seqlen, num_heads, head_dim), bf16 compute.

Forward-mode autodiff (torch.func.jvp) routes through _FlashJVPAttn.jvp,
which calls the CuTeDSL kernel in models/flash_jvp_kernel.py. The tangent
output is only ever used inside a stop-gradient in the MeanFlow loss, so no
reverse-mode graph through the JVP path is required.
"""

import torch

from flash_attn.cute.interface import _flash_attn_fwd, _flash_attn_bwd


def _unwrap_functorch_alias(t):
    """Peel functorch transform wrappers WITHOUT any copy: the returned raw
    tensor aliases the same storage, so in-place kernel writes are visible
    through the wrapped tensor. Not valid for nested transforms over this op
    (we only use a single jvp level)."""
    while torch._C._functorch.is_functorch_wrapped_tensor(t):
        t = torch._C._functorch.get_unwrapped(t)
    return t


def _unwrap_functorch(t):
    """Peel functorch transform wrappers so the raw CUDA tensor (with storage)
    can cross the DLPack boundary into the CuTeDSL kernel. Values are
    unchanged; only the transform bookkeeping is removed. Not valid for
    nested transforms over this op (we only use a single jvp level)."""
    return _unwrap_functorch_alias(t).contiguous()


class _FlashJVPAttn(torch.autograd.Function):
    @staticmethod
    def forward(q, k, v):
        out, lse, _, _ = _flash_attn_fwd(q, k, v, return_lse=True)
        return out, lse

    @staticmethod
    def setup_context(ctx, inputs, output):
        q, k, v = inputs
        out, lse = output
        ctx.mark_non_differentiable(lse)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.save_for_forward(q, k, v)

    @staticmethod
    def backward(ctx, dout, dlse):
        q, k, v, out, lse = ctx.saved_tensors
        if dout is None:
            dout = torch.zeros_like(out)
        dq, dk, dv = _flash_attn_bwd(
            q, k, v, out, dout, lse,
            None,   # softmax_scale -> default 1/sqrt(head_dim)
            False,  # causal
            0.0,    # softcap
        )
        return dq, dk, dv

    @staticmethod
    def jvp(ctx, tq, tk, tv):
        from models.flash_jvp_kernel import flash_attn_jvp_func

        from torch._functorch.pyfunctorch import temporarily_clear_interpreter_stack

        q, k, v = ctx.saved_for_forward
        q, k, v = _unwrap_functorch(q), _unwrap_functorch(k), _unwrap_functorch(v)
        tq = None if tq is None else _unwrap_functorch(tq.to(q.dtype))
        tk = None if tk is None else _unwrap_functorch(tk.to(k.dtype))
        tv = None if tv is None else _unwrap_functorch(tv.to(v.dtype))
        # Drop below every functorch interpreter so tensor factories inside the
        # kernel wrapper allocate real storage that can cross DLPack.
        with temporarily_clear_interpreter_stack():
            if tq is None:
                tq = torch.zeros_like(q)
            if tk is None:
                tk = torch.zeros_like(k)
            if tv is None:
                tv = torch.zeros_like(v)
            _, t_out, _ = flash_attn_jvp_func(q, k, v, tq, tk, tv)
        return t_out, None


class _FusedFlashJVPAttn(torch.autograd.Function):
    """Single-kernel forward+JVP path, used only under torch.func.jvp.

    forward() allocates EMPTY o/lse buffers and does not launch anything; the
    fused CuTeDSL kernel invoked from jvp() fills them in place (the unwrapped
    saved outputs alias the same storage). functorch's custom_function_call
    for the jvp transform runs forward -> setup_context -> jvp before the
    outputs become visible to the caller, so the buffers are filled before
    anyone reads them. This removes the redundant FA4 forward launch that
    _FlashJVPAttn performs in the jvp pass.
    """

    @staticmethod
    def forward(q, k, v):
        from torch._functorch.pyfunctorch import temporarily_clear_interpreter_stack

        # Lazy: buffers filled by jvp() via the fused kernel before anyone
        # reads them. Only ever invoked under torch.func.jvp (see
        # flash_jvp_attention dispatch).
        with temporarily_clear_interpreter_stack():
            batch, seqlen, num_head, _ = q.shape
            o = torch.empty_like(q, memory_format=torch.contiguous_format)
            lse = torch.empty(
                batch, num_head, seqlen, dtype=torch.float32, device=q.device
            )
        return o, lse

    @staticmethod
    def setup_context(ctx, inputs, output):
        q, k, v = inputs
        o, lse = output
        ctx.mark_non_differentiable(lse)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.save_for_forward(q, k, v, o, lse)

    @staticmethod
    def jvp(ctx, tq, tk, tv):
        from models.flash_jvp_kernel import flash_attn_jvp_func

        from torch._functorch.pyfunctorch import temporarily_clear_interpreter_stack

        q, k, v, o, lse = ctx.saved_for_forward
        q, k, v = _unwrap_functorch(q), _unwrap_functorch(k), _unwrap_functorch(v)
        # The saved outputs must be unwrapped WITHOUT .contiguous(): the raw
        # tensors alias the buffers returned by forward(), so the in-place
        # kernel write fills the graph outputs.
        o = _unwrap_functorch_alias(o)
        lse = _unwrap_functorch_alias(lse)
        tq = None if tq is None else _unwrap_functorch(tq.to(q.dtype))
        tk = None if tk is None else _unwrap_functorch(tk.to(k.dtype))
        tv = None if tv is None else _unwrap_functorch(tv.to(v.dtype))
        with temporarily_clear_interpreter_stack():
            if tq is None:
                tq = torch.zeros_like(q)
            if tk is None:
                tk = torch.zeros_like(k)
            if tv is None:
                tv = torch.zeros_like(v)
            t_out = torch.empty_like(q, memory_format=torch.contiguous_format)
            flash_attn_jvp_func(q, k, v, tq, tk, tv, out=o, t_out=t_out, lse_out=lse)
        return t_out, None

    @staticmethod
    def backward(ctx, dout, dlse):
        q, k, v, out, lse = ctx.saved_tensors
        # Cheap sanity that jvp() actually filled the lazy buffers.
        assert torch.isfinite(out.view(-1)[0]), "fused JVP buffers never filled"
        if dout is None:
            dout = torch.zeros_like(out)
        dq, dk, dv = _flash_attn_bwd(
            q, k, v, out, dout, lse,
            None,   # softmax_scale -> default 1/sqrt(head_dim)
            False,  # causal
            0.0,    # softcap
        )
        return dq, dk, dv


def flash_jvp_attention(q, k, v):
    """Drop-in attn_impl for the video DiT: (B, S, H, D) any float dtype.

    Inside a torch.func transform (this project only uses jvp) dispatch to the
    fused single-launch op; otherwise use the stock FA4-forward op.
    """
    orig_dtype = q.dtype
    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    v = v.to(torch.bfloat16)
    if torch._C._functorch.is_functorch_wrapped_tensor(q):
        out, _ = _FusedFlashJVPAttn.apply(q, k, v)
    else:
        out, _ = _FlashJVPAttn.apply(q, k, v)
    return out.to(orig_dtype)


def sdpa_math_attention(q, k, v):
    """Reference implementation: math SDPA, jvp-compatible, slow."""
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, S, D)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2)
