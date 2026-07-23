# Triton Transformer-Block Forward-Mode JVP

## Settings

- Files: `models/triton_block_jvp.py` (function `triton_block_jvp`, module `TritonBlockJVP`), `tests/test_triton_block_jvp.py`

- Hardware: NVIDIA GeForce RTX 5070 Ti (compute capability 12.0, 16 GB), software: `/opt/miniconda3/bin/python3`, torch 2.11.0+cu130, Triton 3.6.0

- Block: pre-LN (pre-LayerNorm), bias-free, bf16 io with fp32 accumulate: x → RMSNorm1 → QKV (query-key-value) linear → multi-head attention (head dim 64) → out linear → residual add → RMSNorm2 → SwiGLU MLP (gate/up d→f with f = 8d/3 rounded to a multiple of 64, down f→d) → residual add

- JVP (Jacobian-vector product) semantics are MeanFlow-style: tangent w.r.t. the input x only, all parameter tangents zero, so primal and tangent pass through every linear with the SAME weight

- Every linear therefore runs as ONE cuBLAS GEMM (general matrix multiplication) on the stacked batch [x; dx] of shape (2B, S, D); gate and up projections are fused into one (2F, D) weight, giving 4 GEMMs total per block JVP

- Triton kernels handle the three nonlinear ops, each producing primal and tangent in a single fused pass: RMSNorm JVP (one read of x, dx), flash-attention JVP (online softmax, o and do in one kernel, 6 tensor-core dots per tile), SwiGLU JVP (elementwise, fp32 math inside)

- All intermediate buffers are preallocated stacked (2B, S, ·) tensors; the Triton kernels write primal into the first half and tangent into the second half so the following stacked GEMM needs no copy or concatenation

- Flash kernel math per (m, n) tile, with softmax scale $c = 1/\sqrt{d_h}$, running row max $m$, row sum $\ell$, tangent row sum $\mu$ (the $m$-shift cancels exactly in the JVP, so streaming is exact):

$$
\begin{equation}
\begin{aligned}
S &= c\, Q K^{\top}, \quad
dS = c\, (dQ\, K^{\top} + Q\, dK^{\top}), \\
\tilde{P} &= \exp(S - m), \quad
T = \tilde{P} \odot dS, \\
\ell &= \textbf{rowsum}(\tilde{P}), \quad
\mu = \textbf{rowsum}(T), \\
\mathrm{acc}_O &\mathrel{+}= \tilde{P} V, \quad
\mathrm{acc}_{dO} \mathrel{+}= T V + \tilde{P}\, dV, \\
O &= \mathrm{acc}_O / \ell, \quad
dO = \mathrm{acc}_{dO} / \ell - (\mu / \ell) \odot O
\end{aligned}
\end{equation}
$$

- RMSNorm JVP per row, with gain $w$ and $\epsilon = 10^{-6}$; SwiGLU JVP elementwise with $\sigma$ the sigmoid:

$$
\begin{equation}
\begin{aligned}
r &= \sqrt{\textbf{mean}(x^2) + \epsilon}, \quad
y = w\, x / r, \\
dy &= w \left( dx / r - x\, \textbf{mean}(x\, dx) / r^3 \right), \\
h &= \textbf{silu}(g)\, u, \quad
dh = \textbf{silu}'(g)\, dg\, u + \textbf{silu}(g)\, du, \\
\textbf{silu}'(g) &= \sigma(g) \left( 1 + g\, (1 - \sigma(g)) \right)
\end{aligned}
\end{equation}
$$

- Flash kernel implementation: exp2-based online softmax, arbitrary input strides (q, k, v, dq, dk, dv are strided views of the single stacked QKV GEMM output, no transpose copies), autotuned over tile configs {(64,64), (128,64), (64,128), (128,128)} x warps x stages with an EVEN_S specialization that drops all bounds masks when S is a multiple of 128; autotune selects BLOCK_M = 64, BLOCK_N = 64, num_warps = 4, num_stages = 2 (same optimum as the CuTeDSL tile sweep in `records/flash_jvp_kernel.md`)

## Correctness

- Reference: pure PyTorch block under `torch.func.jvp`, fp32, math SDPA (scaled dot-product attention) backend forced via `sdpa_kernel([SDPBackend.MATH])`, on the same bf16-representable weights and inputs; threshold: relative Frobenius error $\le 3 \times 10^{-2}$ on y and dy, bf16 run

| Shape (B, S, D, H) | y rel Fro | y max abs | dy rel Fro | dy max abs |
|---|---|---|---|---|
| (2, 512, 768, 12) | 2.464e-03 | 2.781e-02 | 2.570e-03 | 2.381e-02 |
| (1, 1000, 768, 12) | 2.463e-03 | 2.863e-02 | 2.567e-03 | 3.109e-02 |
| (2, 4096, 512, 8) | 2.298e-03 | 2.143e-02 | 2.335e-03 | 2.038e-02 |

- (1, 1000, ...) exercises a sequence length that is not a multiple of the tile size; `TritonBlockJVP` module output is bit-exact versus the functional `triton_block_jvp` path; all 4 pytest cases pass

## Benchmark

- bf16, CUDA events, 30 iterations after 5 warmup, versus `torch.func.jvp` over the eager PyTorch block (math SDPA is the only jvp-capable PyTorch attention path) and versus `torch.compile` of the same jvp closure (compile succeeded, no graph error)

| Shape (B, S, D, H) | Triton (ms) | Eager jvp (ms) | Speedup | Compiled jvp (ms) | Speedup |
|---|---|---|---|---|---|
| (2, 1024, 768, 12) | 0.931 | 6.219 | 6.7x | 2.986 | 3.2x |
| (2, 4096, 768, 12) | 6.271 | 81.667 | 13.0x | 33.161 | 5.3x |
| (1, 8192, 768, 12) | 9.554 | OOM | n/a | 61.482 | 6.4x |

- (1, 8192, 768, 12) eager baseline: CUDA out of memory on the 16 GB card (math-SDPA jvp materializes multiple 12 x 8192 x 8192 score tensors, including a 3 GiB fp32 allocation, exceeding 15.47 GiB); the Triton path runs the same shape in under 3 GiB, so the eager speedup there is unbounded on this hardware

- Component breakdown at (1, 8192, 768, 12): flash-attention JVP 6.59 ms, the four stacked cuBLAS GEMMs 2.50 ms total, RMSNorm/SwiGLU/residual ops about 0.4 ms

- Wall at S = 1024 (6.7x): attention is a small fraction of the block there; the remaining work is GEMMs that the eager baseline also executes at near-peak cuBLAS throughput on both primal and tangent, so the attainable ratio is bounded by the non-attention fraction

- Wall at S = 8192 versus the compiled baseline (6.4x): the flash JVP kernel executes 618 GFLOP (6 dots x 12 heads x 8192^2 x 64) in 6.59 ms, about 94 TFLOPS effective, near the practical bf16 tensor-core ceiling for Triton on this consumer Blackwell part; the eager 10x target is met at S = 4096 (13.0x) and the eager baseline cannot run at all at S = 8192

- Repro: `/opt/miniconda3/bin/python3 tests/test_triton_block_jvp.py` (correctness then benchmark), or `python3 -m pytest tests/test_triton_block_jvp.py` for correctness only
