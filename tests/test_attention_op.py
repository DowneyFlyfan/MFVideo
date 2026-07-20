"""Test flash_jvp_attention: reverse-mode grads and forward-mode jvp vs math SDPA."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F


def sdpa_ref(q, k, v):
    # (B, S, H, D) fp32 math reference (math backend: only one with forward AD)
    q, k, v = (t.transpose(1, 2).float() for t in (q, k, v))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2)


def report(name, a, b, tol):
    err = (a.float() - b.float()).abs().max().item()
    rel = (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-12)
    status = "OK" if err <= tol else "FAIL"
    print(f"{name}: max_abs={err:.3e} rel_fro={rel:.3e} [{status}]")
    return err <= tol


def main():
    from models.attention_op import flash_jvp_attention

    torch.manual_seed(0)
    device = "cuda"
    ok = True

    for (B, S, H, D) in [(2, 768, 6, 64), (1, 212, 6, 64)]:
        q, k, v = (torch.randn(B, S, H, D, device=device) * 0.5 for _ in range(3))
        tq, tk, tv = (torch.randn(B, S, H, D, device=device) * 0.5 for _ in range(3))

        # --- reverse mode ---
        q1, k1, v1 = (t.clone().requires_grad_(True) for t in (q, k, v))
        out = flash_jvp_attention(q1, k1, v1)
        out.backward(tq)  # arbitrary cotangent
        q2, k2, v2 = (t.clone().requires_grad_(True) for t in (q, k, v))
        ref = sdpa_ref(q2, k2, v2)
        ref.backward(tq)

        ok &= report(f"({B},{S},{H},{D}) fwd", out, ref, 2e-2)
        ok &= report(f"({B},{S},{H},{D}) dq ", q1.grad, q2.grad, 2e-2)
        ok &= report(f"({B},{S},{H},{D}) dk ", k1.grad, k2.grad, 2e-2)
        ok &= report(f"({B},{S},{H},{D}) dv ", v1.grad, v2.grad, 2e-2)

        # --- forward mode (jvp) ---
        out_j, t_out = torch.func.jvp(flash_jvp_attention, (q, k, v), (tq, tk, tv))
        ref_j, t_ref = torch.func.jvp(sdpa_ref, (q, k, v), (tq, tk, tv))
        ok &= report(f"({B},{S},{H},{D}) jvp o ", out_j, ref_j, 2e-2)
        ok &= report(f"({B},{S},{H},{D}) jvp to", t_out, t_ref, 2e-2)

    print("ALL PASS" if ok else "FAILURES PRESENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
