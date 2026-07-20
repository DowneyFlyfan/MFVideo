# improved MeanFlow on Video — Demo Training Round (RTX 5070 Ti)

- Date: 2026-07-19 (fusion update 2026-07-20)

## Goal

- Apply improved MeanFlow (iMF) to video using the Wan2.1 VAE (Variational AutoEncoder) for latents, with the JVP (Jacobian-Vector Product, forward-mode autodiff) pass routed through a custom CuTeDSL FlashAttention kernel, and run 1 round of training.

## Environment

- GPU: NVIDIA GeForce RTX 5070 Ti, 16 GB, compute capability sm_120 (Blackwell GeForce)

- PyTorch 2.11.0+cu130, CUDA 13.3, python 3.14 (/opt/miniconda3), project venv `.venv` (`uv venv --system-site-packages -p /opt/miniconda3/bin/python`)

- flash-attn-4 4.0.0b21 (CuTeDSL sources), nvidia-cutlass-dsl 4.6.0.dev0, diffusers 0.36 (venv, with `kernels<0.15` pin to fix import crash against transformers 5.7.0)

## Components

- **CuTeDSL fused forward+JVP FlashAttention kernel** — [models/flash_jvp_kernel.py](../models/flash_jvp_kernel.py), class `FlashAttentionJvpSm120` (subclasses FlashAttention-4's `FlashAttentionForwardSm80`, SM80 MMA paths on sm_120). Details in [flash_jvp_kernel.md](flash_jvp_kernel.md).

- Symbols: $Q, K, V$ = query/key/value tiles; $\dot Q, \dot K, \dot V$ = their tangents; $S$ = score tile; $\tilde P$ = unnormalized softmax probabilities; $m$ = running row max (cancels exactly in the JVP); $\ell$ = softmax normalizer; $O, \dot O$ = output and its tangent; $c = 1/\sqrt{D}$ = softmax scale with $D$ = head dimension; $\odot$ = elementwise product.

$$
\begin{equation}
\begin{aligned}
S &= c\,QK^\top, \qquad
\dot S = c\,(\dot Q K^\top + Q \dot K^\top), \qquad
\tilde P = e^{S - m}, \\
\ell &= \mathrm{rowsum}(\tilde P), \qquad
O = \frac{\tilde P V}{\ell}, \\
\dot O &= \frac{(\tilde P \odot \dot S)\,V + \tilde P\,\dot V}{\ell} - \\
&\qquad \frac{\mathrm{rowsum}(\tilde P \odot \dot S)}{\ell}\odot O
\end{aligned}
\end{equation}
$$

- 6 GEMMs (General Matrix Multiplications) per tile: $QK^\top$, $\dot Q K^\top$, $Q\dot K^\top$, $\tilde P V$, $(\tilde P\odot\dot S)V$, $\tilde P \dot V$; online-softmax rescaling applied jointly to the accumulators of $O$, $\dot O$, $\mathrm{rowsum}(\tilde P\odot\dot S)$ and $\ell$; tiles (tile_m, tile_n) = (64, 64), 128 threads, 48 KB shared memory, bf16 in / fp32 accumulate.

- **Attention autograd op** — [models/attention_op.py](../models/attention_op.py): `torch.autograd.Function` with primal forward = FlashAttention-4 `_flash_attn_fwd`, reverse backward = FlashAttention-4 `_flash_attn_bwd`, forward-mode `jvp` = CuTeDSL kernel. functorch integration required `ctx.save_for_forward`, `ctx.mark_non_differentiable(lse)`, unwrapping functorch tensor wrappers, and allocating kernel outputs under `temporarily_clear_interpreter_stack()` (tensor factories inside an active jvp interpreter otherwise return storage-less wrapped tensors that cannot cross DLPack).

- **Video DiT (Diffusion Transformer)** — [models/imf_dit_video.py](../models/imf_dit_video.py): PyTorch port of the JAX `imfDiT`, extended to video: Conv3d patchify (1,2,2), 3D axial RoPE (Rotary Position Embedding, head_dim 64 split t/h/w = 16/24/24), in-context conditioning tokens (class / CFG scale $\omega$ / CFG interval / $h = t-r$), RMSNorm QK-norm, SwiGLU MLP with ratio 8/3, zero-init residual vector gates, shared trunk + separate u/v heads, zero-init final layers.

- **iMF loss** — [imf_video_pt.py](../imf_video_pt.py): faithful port of `imf.py`: logit-normal $t,r$ with $P_{\textbf{mean}}=-0.4$, $P_{\textbf{std}}=1.0$; data proportion 0.5; CFG (Classifier-Free Guidance) scale power sampling with $s_{\max}=7$; CFG interval sampling; guidance function under `no_grad`; conditional dropout 0.1; `torch.func.jvp` with tangents $(v_c, 1, 0)$; compound target $V = u + (t-r)\,\mathrm{sg}(\mathrm{d}u/\mathrm{d}t)$ where $\mathrm{sg}$ = stop-gradient; adaptive weighting $(\textbf{loss}+0.01)^{1}$ detached.

- **Data** — 8 synthetic moving-pattern clips, shape (3, 9, 128, 128) in $[-1,1]$, encoded by `AutoencoderKLWan` (Wan-AI/Wan2.1-T2V-1.3B-Diffusers, bf16) → latents (8, 16, 3, 16, 16), normalized per channel by config `latents_mean`/`latents_std`; latent std after normalization 0.972. Sequence length: 192 video tokens + 20 conditioning tokens = 212.

## Kernel correctness (bf16, vs fp32 `torch.func.jvp` over math SDPA)

| shape (B,S,H,D) | O max-abs / rel-Fro | tangent max-abs / rel-Fro |
|---|---|---|
| (2,768,6,64) | 9.9e-4 / 2.2e-3 | 3.0e-3 / 2.3e-3 |
| (1,212,6,64) | 1.9e-3 / 2.1e-3 | 4.1e-3 / 2.3e-3 |
| (2,1024,8,64) | 1.2e-3 / 2.2e-3 | 3.1e-3 / 2.3e-3 |

- Integration test ([tests/test_attention_op.py](../tests/test_attention_op.py)): forward, dq/dk/dv reverse grads, and jvp all within 6e-4 max-abs of fp32 reference — ALL PASS.

- Kernel benchmark vs naive `torch.func.jvp` over math-backend SDPA (bf16): (2,4096,8,64): 2.21 ms vs 50.7 ms (23.0×); (2,768,6,64): 0.087 ms vs 1.31 ms (15.2×).

## Training settings

- Model: `imf_dit_video_S` — hidden 384, depth 8 (+4 aux head depth), heads 6, head_dim 64, 22.33 M parameters, fp32 master weights, bf16 attention compute

- Optimizer: AdamW, lr 1e-4, weight decay 0, grad clip 1.0; batch 4; 10 steps; num_classes 10 (random labels); seed 0

- Command: `.venv/bin/python train_demo.py --steps 10 [--attn sdpa]`

## Results

- Per-step verification counters (1 step, batch 4): 12 CuTeDSL JVP kernel calls (8 shared + 4 u-head layers), 36 FlashAttention-4 forwards (12 jvp-pass primal + 24 guidance-pass), 12 FlashAttention-4 backwards.

| run | step 0 loss_u | step 9 loss_u | s/step (steady) | 10-step wall | peak VRAM |
|---|---|---|---|---|---|
| flash_jvp (CuTeDSL) | 1.9690 | 1.8407 | 0.08 | 4.4 s (incl. 3.7 s first-step kernel compile) | 1.66 GiB |
| sdpa math baseline | 1.9690 | 1.8407 | 0.08 | 1.2 s | 1.74 GiB |

- Total loss stays ≈ 2.0000 by construction (adaptive weighting normalizes each of loss_u/loss_v toward 1); unweighted loss_u decreased 1.9690 → 1.8407, loss_v 1.9690 → 1.8399; all losses and gradients finite; grad_norm range 0.74–1.25.

- Loss histories: [train_demo_history_flash_jvp.json](train_demo_history_flash_jvp.json), [train_demo_history_sdpa.json](train_demo_history_sdpa.json). Losses of the two runs match to 4 printed decimals (bf16-vs-fp32 attention differences below print precision at this scale; zero-init gates damp propagation).

- At sequence length 212 the two attention paths have equal steady-state step time; kernel-level speedup (15–23×) manifests at longer sequences (see benchmark above).

## Single-Launch JVP Fusion

- Previously, under `torch.func.jvp` each attention layer launched TWO kernels for the primal: the FlashAttention-4 stock forward in `forward()` (for output $O$ and logsumexp $L$ needed by the reverse backward), plus a redundant primal recompute inside the CuTeDSL forward+JVP kernel invoked from `jvp()` (whose $O$ was discarded).

- Approach shipped: **lazy-buffer fusion** (the first design; the dual-unpack fallback was not needed). The CuTeDSL kernel gained an fp32 LSE output tensor with FA4 stock layout $(B, H, S)$ and natural-log semantics (see [flash_jvp_kernel.md](flash_jvp_kernel.md)), plus preallocated-buffer arguments `out` / `t_out` / `lse_out` in `flash_attn_jvp_func`.

- New `_FusedFlashJVPAttn(torch.autograd.Function)` in [models/attention_op.py](../models/attention_op.py): `forward()` only allocates EMPTY $O$ and $L$ buffers under `temporarily_clear_interpreter_stack()` (no kernel launch); `jvp()` unwraps the saved outputs WITHOUT `.contiguous()` so the raw tensors alias the graph outputs' storage, then a single fused kernel launch writes $O$, $\dot O$, and $L$ in place; `backward()` runs the FlashAttention-4 backward on the saved $q, k, v, O, L$ with a cheap finiteness assert on $O$.

- Correctness of the laziness relies on functorch's `custom_function_call` for the jvp transform running `forward` → `setup_context` → `jvp` before outputs become visible, and on external (DLPack) in-place writes not bumping PyTorch tensor version counters; aliasing was verified by tests (nonzero, correct outputs after jvp).

- Dispatch in `flash_jvp_attention`: `torch._C._functorch.is_functorch_wrapped_tensor(q)` selects `_FusedFlashJVPAttn` inside a `torch.func` transform (this project only uses jvp); guidance passes and plain forwards keep the stock `_FlashJVPAttn` path; bf16 casting unchanged.

- Kernel-launch accounting ([tests/test_fusion_count.py](../tests/test_fusion_count.py), one iMF loss step, batch 4, `imf_dit_video_S`, num_classes 10, 12 attention layers): fused CuTeDSL kernel calls = 12, FlashAttention-4 forwards = 24 (guidance passes only, previously 36), FlashAttention-4 backwards = 12 — exactly one kernel launch per attention layer in the jvp pass.

- Micro-benchmark of the jvp-pass attention (bf16, 100 iterations after 20 warmup): (4, 212, 6, 64): fused single launch 0.021 ms versus old FA4-fwd + jvp-kernel double launch 0.033 ms (1.58x); (2, 4096, 8, 64): 2.219 ms versus 3.027 ms (1.36x).

- Training re-run `.venv/bin/python train_demo.py --steps 10`: completes, all losses finite; loss_u 1.9690 → 1.8407 identical to the pre-fusion run at 4 printed decimals; steady-state 0.08 s/step, 10-step wall 4.4 s including 3.71 s first-step kernel compile, peak VRAM 1.66 GiB (previous flash_jvp run: 0.08 s/step, 4.4 s wall including 3.7 s compile) — at sequence length 212 the saved launch is below step-time print precision; the gain shows in the micro-benchmark above.

- All prior tests pass unchanged with identical thresholds: `tests/test_attention_op.py` (forward, dq/dk/dv, jvp — ALL PASS), `tests/test_flash_jvp.py` (o / t_o errors unchanged, LSE max absolute error 9.537e-07 versus stock forward), `tests/test_model_pt.py` (finite grads on 186 parameter tensors).

- History of the fused run: [train_demo_history_flash_jvp_fused.json](train_demo_history_flash_jvp_fused.json)

## Files

- Kernel: `models/flash_jvp_kernel.py`; op: `models/attention_op.py`; model: `models/imf_dit_video.py`; loss: `imf_video_pt.py`; training: `train_demo.py`

- Tests: `tests/test_flash_jvp.py`, `tests/test_attention_op.py`, `tests/test_model_pt.py`, `tests/test_fusion_count.py`

- Kernel record: `records/flash_jvp_kernel.md`
