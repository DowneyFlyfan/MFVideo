"""Demo training round: improved MeanFlow on video latents from Wan2.1 VAE.

Pipeline: synthetic demo videos -> WanVAE encode -> latents -> iMF loss with
CuTeDSL flash-attention JVP kernel -> 1 training round (N optimizer steps) on
RTX 5070 Ti.

Run inside .venv: .venv/bin/python train_demo.py
"""

import argparse
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def make_demo_videos(num_clips, num_frames, height, width, device):
    """Synthetic moving-pattern videos in [-1, 1], shape (N, 3, T, H, W)."""
    torch.manual_seed(42)
    ys = torch.linspace(-1, 1, height, device=device).view(1, 1, height, 1)
    xs = torch.linspace(-1, 1, width, device=device).view(1, 1, 1, width)
    ts = torch.arange(num_frames, device=device).view(1, num_frames, 1, 1)
    clips = []
    for i in range(num_clips):
        phase = 2 * math.pi * i / num_clips
        freq = 2.0 + i % 3
        r = torch.sin(freq * math.pi * (xs + 0.05 * ts) + phase)
        g = torch.cos(freq * math.pi * (ys - 0.05 * ts) + phase)
        b = torch.sin(freq * math.pi * (xs + ys + 0.1 * ts))
        clips.append(
            torch.cat(
                [
                    r.expand(1, num_frames, height, width),
                    g.expand(1, num_frames, height, width),
                    b.expand(1, num_frames, height, width),
                ],
                dim=0,
            )
        )
    return torch.stack(clips)  # (N, 3, T, H, W)


@torch.no_grad()
def encode_latents(videos, vae, batch=2):
    """WanVAE encode + normalization to ~unit variance latents."""
    mean = torch.tensor(vae.config.latents_mean, device=videos.device).view(
        1, -1, 1, 1, 1
    )
    inv_std = 1.0 / torch.tensor(vae.config.latents_std, device=videos.device).view(
        1, -1, 1, 1, 1
    )
    outs = []
    for i in range(0, videos.shape[0], batch):
        chunk = videos[i : i + batch].to(vae.dtype)
        latent = vae.encode(chunk).latent_dist.sample()
        outs.append(((latent - mean) * inv_std).float())
    return torch.cat(outs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-clips", type=int, default=8)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--attn", choices=["flash_jvp", "sdpa"], default="flash_jvp")
    args = parser.parse_args()

    device = "cuda"
    torch.manual_seed(0)

    # ---- WanVAE encode ----
    from diffusers import AutoencoderKLWan

    print("loading WanVAE ...", flush=True)
    vae = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae", torch_dtype=torch.bfloat16
    ).to(device)
    vae.eval()

    videos = make_demo_videos(args.num_clips, args.frames, args.size, args.size, device)
    latents = encode_latents(videos, vae, batch=2)
    del vae
    torch.cuda.empty_cache()
    print(f"latents: {tuple(latents.shape)} std={latents.std().item():.3f}", flush=True)

    labels_all = torch.randint(0, args.num_classes, (args.num_clips,), device=device)

    # ---- model + loss ----
    from models.imf_dit_video import imf_dit_video_S
    from models.attention_op import flash_jvp_attention, sdpa_math_attention
    from imf_video import IMFVideoLoss

    attn_impl = flash_jvp_attention if args.attn == "flash_jvp" else sdpa_math_attention
    net = imf_dit_video_S(num_classes=args.num_classes, attn_impl=attn_impl).to(device)
    num_params = sum(p.numel() for p in net.parameters())
    print(f"model params: {num_params / 1e6:.2f}M attn={args.attn}", flush=True)

    imf_loss = IMFVideoLoss(net, num_classes=args.num_classes)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)

    # ---- 1 training round ----
    history = []
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, args.num_clips, (args.batch_size,))
        batch_latents = latents[idx]
        batch_labels = labels_all[idx]

        t_step = time.time()
        loss, dict_losses = imf_loss(batch_latents, batch_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()

        entry = {
            "step": step,
            "loss": loss.item(),
            "loss_u": dict_losses["loss_u"].item(),
            "loss_v": dict_losses["loss_v"].item(),
            "grad_norm": grad_norm.item(),
            "time_s": time.time() - t_step,
        }
        history.append(entry)
        print(
            f"step {step:3d} loss={entry['loss']:.4f} loss_u={entry['loss_u']:.4f} "
            f"loss_v={entry['loss_v']:.4f} grad_norm={entry['grad_norm']:.3f} "
            f"({entry['time_s']:.2f}s)",
            flush=True,
        )
        assert math.isfinite(entry["loss"]), "non-finite loss"

    peak_mem = torch.cuda.max_memory_allocated() / 2**30
    total_time = time.time() - t_start
    print(
        f"done: {args.steps} steps in {total_time:.1f}s, peak VRAM {peak_mem:.2f} GiB",
        flush=True,
    )

    os.makedirs("records", exist_ok=True)
    import json

    with open("records/train_demo_history.json", "w") as f:
        json.dump(
            {"args": vars(args), "history": history, "peak_vram_gib": peak_mem},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
