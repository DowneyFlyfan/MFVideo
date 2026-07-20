# Fused FlashAttention Forward + JVP Kernel (SM120)

## Settings

- File: `models/flash_jvp_kernel.py`, class `FlashAttentionJvpSm120`, wrapper `flash_attn_jvp_func`

- Hardware: NVIDIA GeForce RTX 5070 Ti (compute capability 12.0)

- Software: `/opt/miniconda3/bin/python3`, torch 2.11.0+cu130, flash-attn 4.0.0b21 (CuTeDSL sources), nvidia-cutlass-dsl 4.6.0.dev0

- Kernel configuration: tile_m = 64, tile_n = 64, num_stages = 1, num_threads = 128 (4 warps), fp32 accumulate, Q_in_regs = False, non-causal, fixed sequence length ($S_q = S_k$), no grouped-query attention, no variable-length batching, no dropout

- Shared memory: 6 tiles ($Q$, $K$, $V$, $dQ$, $dK$, $dV$), 48 KB of 99 KB

- 6 GEMMs (general matrix multiplications) per $(m, n)$ tile: $QK^{\top}$, $dQ\,K^{\top}$, $Q\,dK^{\top}$, $\tilde{P}V$, $TV$, $\tilde{P}\,dV$

- Streaming per-row state and final combination, where $s$ is the softmax scale, $m$ the running row max, $\ell$ the running row sum, $\tilde{\mu}$ the tangent row sum, $\mathrm{acc}_O$ accumulates $\tilde{P}V$, and $\mathrm{acc}_{dO}$ accumulates $TV + \tilde{P}\,dV$:

$$
\begin{equation}
\begin{aligned}
\tilde{P} &= \exp\left(s\,(QK^{\top} - m)\right), \\
T &= \tilde{P} \odot \left(s\,(dQ\,K^{\top} + Q\,dK^{\top})\right), \\
\ell &= \textbf{rowsum}(\tilde{P}), \quad
\tilde{\mu} = \textbf{rowsum}(T), \\
O &= \mathrm{acc}_O / \ell, \\
dO &= \mathrm{acc}_{dO} / \ell -
(\tilde{\mu} / \ell) \odot (\mathrm{acc}_O / \ell)
\end{aligned}
\end{equation}
$$

- Softmax scale default $1/\sqrt{D}$; inputs bf16 or fp16 (fp32 cast to bf16); compile cache keyed by (dtype, head_dim), all shapes dynamic

## Logsumexp Output

- The kernel additionally writes the softmax logsumexp $L$ (LSE) to a required fp32 output tensor of shape $(B, H, S)$ — the exact allocation layout of the FlashAttention-4 stock forward (`_flash_attn_fwd` with `return_lse=True`), transposed inside the kernel wrapper via mode $[2, 1, 0]$ to $(S, H, B)$ as in the stock `flash_fwd` launch path

- Semantics: `Softmax.finalize` rewrites its running row sum in place to natural-log units using the exp2/log2 scaling trick; with $s$ the softmax scale, $m$ the running row max, and $\ell = \textbf{rowsum}(\tilde{P})$, the stored value per row is

$$
\begin{equation}
\begin{aligned}
L &= \ln 2 \cdot \left( m \, s \log_2 e + \log_2 \ell \right) \\
&= s\, m + \ln \ell = \ln \sum_j \exp\left(s\, S_j\right)
\end{aligned}
\end{equation}
$$

- Rows with $\ell = 0$ or NaN store $L = -\infty$, matching the stock forward; this is exactly the natural-log LSE consumed by the FA4 backward `_flash_attn_bwd`

- Preallocated output buffers: `flash_attn_jvp_func(q, k, v, tq, tk, tv, softmax_scale=None, out=None, t_out=None, lse_out=None)` writes into `out` $(B, S, H, D)$, `t_out` $(B, S, H, D)$, `lse_out` $(B, H, S)$ fp32 when given (shape/dtype/contiguity asserted), allocates otherwise; returns `(o, t_o, lse)`

- LSE correctness (`tests/test_flash_jvp.py`, versus stock `_flash_attn_fwd(..., return_lse=True)` on identical inputs, tolerance $10^{-3}$): max absolute error 9.537e-07 on all three test shapes; in-place buffer fill bit-exact versus allocation path

## Test Results

- Test file: `tests/test_flash_jvp.py`, bf16, versus `torch.func.jvp` over fp32 `scaled_dot_product_attention` (math backend); thresholds: max absolute error $\le 2\times 10^{-2}$, relative Frobenius error $\le 2\times 10^{-2}$

| Shape (B, S, H, D) | O max abs | O rel Fro | dO max abs | dO rel Fro |
|---|---|---|---|---|
| (2, 768, 6, 64) | 9.897e-04 | 2.205e-03 | 3.041e-03 | 2.326e-03 |
| (1, 212, 6, 64) | 1.941e-03 | 2.130e-03 | 4.126e-03 | 2.318e-03 |
| (2, 1024, 8, 64) | 1.200e-03 | 2.232e-03 | 3.103e-03 | 2.316e-03 |

- fp16 sanity (2, 768, 6, 64): O max abs 1.468e-04, dO max abs 5.147e-04

## Benchmark

- bf16, 30 iterations after 10 warmup, versus `torch.func.jvp` over `scaled_dot_product_attention` math backend

| Shape | Kernel (ms) | Naive JVP (ms) | Speedup |
|---|---|---|---|
| (2, 4096, 8, 64) | 2.205 | 50.736 | 23.0x |
| (2, 768, 6, 64) | 0.087 | 1.312 | 15.2x |

- Tile sweep at (2, 4096, 8, 64): (64,64) 2.206 ms; (128,64) 4.090 ms; (64,128) 3.613 ms; (128,128) 5.318 ms; all tile configurations numerically correct
