"""Smoke test for the PyTorch video imfDiT model and improved MeanFlow loss.

Run: /opt/miniconda3/bin/python3 tests/test_model_pt.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from imf_video import IMFVideoLoss
from models.imf_dit_video import imf_dit_video_S


def main():
    assert torch.cuda.is_available(), "CUDA device required"
    device = torch.device("cuda")
    torch.manual_seed(0)

    num_classes = 10
    batch, channels, frames, height, width = 2, 16, 3, 16, 16

    net = imf_dit_video_S(num_classes=num_classes).to(device)
    num_params = sum(p.numel() for p in net.parameters())
    print(f"model params: {num_params / 1e6:.2f}M")

    latents = torch.randn(batch, channels, frames, height, width, device=device)
    labels = torch.randint(0, num_classes, (batch,), device=device)

    # --- plain forward shape check ---
    t = torch.rand(batch, device=device)
    u, v = net(latents, t, t, 1 + t, 0 * t, 1 + 0 * t, labels)
    assert u.shape == latents.shape, u.shape
    assert v.shape == latents.shape, v.shape
    print("forward shapes OK:", tuple(u.shape))

    # --- torch.func.jvp path with default SDPA-math attention ---
    def u_fn(z, t_in, r_in):
        u_out, v_out = net(
            z, t_in, t_in - r_in, 1 + 0 * t_in, 0 * t_in, 1 + 0 * t_in, labels
        )
        return u_out, v_out

    tangent_z = torch.randn_like(latents)
    r = 0.5 * t
    u_p, du_dt, v_aux = torch.func.jvp(
        u_fn, (latents, t, r), (tangent_z, torch.ones_like(t), torch.zeros_like(t)),
        has_aux=True,
    )
    assert du_dt.shape == latents.shape
    assert torch.isfinite(u_p).all() and torch.isfinite(du_dt).all()
    assert torch.isfinite(v_aux).all()
    print("torch.func.jvp path OK, |du_dt| mean:", du_dt.abs().mean().item())

    # At init the zero-init gates/final layers make the output (and du_dt)
    # exactly zero; perturb parameters to verify the jvp tangent is nontrivial.
    with torch.no_grad():
        for param in net.parameters():
            param.add_(0.02 * torch.randn_like(param))
    _, du_dt_perturbed, _ = torch.func.jvp(
        u_fn, (latents, t, r), (tangent_z, torch.ones_like(t), torch.zeros_like(t)),
        has_aux=True,
    )
    assert torch.isfinite(du_dt_perturbed).all()
    assert du_dt_perturbed.abs().mean() > 0, "jvp tangent is identically zero"
    print(
        "perturbed jvp |du_dt| mean:", du_dt_perturbed.abs().mean().item()
    )

    # --- full loss forward + backward ---
    loss_module = IMFVideoLoss(net, num_classes=num_classes)
    loss, dict_losses = loss_module(latents, labels)

    assert torch.isfinite(loss), f"non-finite loss: {loss}"
    loss.backward()

    grad_params = 0
    for name, param in net.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"non-finite grad in {name}"
            grad_params += 1
    assert grad_params > 0, "no gradients produced"

    print(f"loss: {loss.item():.6f}")
    for key, value in dict_losses.items():
        print(f"  {key}: {value.item():.6f}")
    print(f"finite grads on {grad_params} parameter tensors")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
