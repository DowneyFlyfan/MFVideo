"""Kernel-launch accounting for one iMF loss step with the fused JVP path.

One improved MeanFlow loss step (batch 4, imf_dit_video_S, num_classes=10)
must launch, per the 12 attention layers (4 shared + 4 u-head + 4 v-head):
  - fused CuTeDSL forward+JVP kernel: 12   (jvp pass, single launch per layer)
  - FlashAttention-4 forward:         24   (guidance passes only; was 36)
  - FlashAttention-4 backward:        12   (reverse pass through the jvp graph)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def main():
    import models.attention_op as attention_op
    import models.flash_jvp_kernel as flash_jvp_kernel
    from models.imf_dit_video import imf_dit_video_S
    from imf_video_pt import IMFVideoLoss

    counts = {"fused_jvp": 0, "fa4_fwd": 0, "fa4_bwd": 0}

    orig_fused = flash_jvp_kernel.flash_attn_jvp_func
    orig_fwd = attention_op._flash_attn_fwd
    orig_bwd = attention_op._flash_attn_bwd

    def counted_fused(*args, **kwargs):
        counts["fused_jvp"] += 1
        return orig_fused(*args, **kwargs)

    def counted_fwd(*args, **kwargs):
        counts["fa4_fwd"] += 1
        return orig_fwd(*args, **kwargs)

    def counted_bwd(*args, **kwargs):
        counts["fa4_bwd"] += 1
        return orig_bwd(*args, **kwargs)

    flash_jvp_kernel.flash_attn_jvp_func = counted_fused
    attention_op._flash_attn_fwd = counted_fwd
    attention_op._flash_attn_bwd = counted_bwd
    try:
        torch.manual_seed(0)
        device = "cuda"
        net = imf_dit_video_S(
            num_classes=10, attn_impl=attention_op.flash_jvp_attention
        ).to(device)
        imf_loss = IMFVideoLoss(net, num_classes=10)

        latents = torch.randn(4, 16, 3, 16, 16, device=device)
        labels = torch.randint(0, 10, (4,), device=device)

        loss, _ = imf_loss(latents, labels)
        loss.backward()
        torch.cuda.synchronize()
        assert torch.isfinite(loss).item(), "non-finite loss"
    finally:
        flash_jvp_kernel.flash_attn_jvp_func = orig_fused
        attention_op._flash_attn_fwd = orig_fwd
        attention_op._flash_attn_bwd = orig_bwd

    print(f"counts: {counts}")
    expected = {"fused_jvp": 12, "fa4_fwd": 24, "fa4_bwd": 12}
    assert counts == expected, f"expected {expected}, got {counts}"
    print("fusion count OK: 12 fused, 24 FA4 fwd (guidance only), 12 FA4 bwd")


def test_fusion_count():
    main()


if __name__ == "__main__":
    main()
