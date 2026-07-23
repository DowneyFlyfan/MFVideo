# Triton Transformer-Block Forward-Mode JVP

## Settings

- Files: `models/triton_block_jvp.py` (function `triton_block_jvp`, module `TritonBlockJVP`), `tests/test_triton_block_jvp.py`

- Hardware: NVIDIA GeForce RTX 5070 Ti (compute capability 12.0, 16 GB), software: `/opt/miniconda3/bin/python3`, torch 2.11.0+cu130, Triton 3.6.0

- Block: pre-LN (pre-LayerNorm), bias-free, bf16 io: x → RMSNorm1 → QKV (query-key-value) linear → multi-head attention (head dim 64) → out linear → residual add → RMSNorm2 → SwiGLU MLP (gate/up d→f with f = 8d/3 rounded to a multiple of 64, down f→d) → residual add

- JVP (Jacobian-vector product) semantics are MeanFlow-style: tangent w.r.t. the input x only, all parameter tangents zero, so primal and tangent pass through every linear with the SAME weight

- Every linear therefore runs as ONE GEMM (general matrix multiplication) on the stacked batch [x; dx] of shape (2B, S, D); gate and up projections are fused into one (2F, D) weight, giving 4 GEMMs total per block JVP

- All 4 GEMMs run in fp8 (float8_e4m3fn) tensor cores via `torch._scaled_mm` with per-tensor scales: each weight is amax-scaled into the fp8 range (descale = amax/448) once and cached in the module; activations are quantized to fp8 at unit scale INSIDE the producing Triton kernel (RMSNorm JVP, flash JVP, SwiGLU JVP store fp8 with a clamp to ±448), so activation quantization costs no extra memory pass; GEMM accumulation stays fp32 in the tensor cores

- Triton kernels handle the three nonlinear ops, each producing primal and tangent in a single fused pass: RMSNorm JVP (one read of x, dx), flash-attention JVP (online softmax, o and do in one kernel), SwiGLU JVP (elementwise, fp32 math inside)

- The flash-attention JVP kernel takes fp16 q/k/v (the QKV `_scaled_mm` emits fp16 directly) and issues every `tl.dot` with `out_dtype=tl.float16` (fp16 MMA accumulation, 2x the fp32-accumulate MMA rate on GeForce); each tile dot is accumulated externally in fp32, so fp16 accumulation only ever spans one tile inner dimension (64 or 128 terms), and all softmax statistics stay fp32

- The two tangent-score dots $dQ\,K^{\top}$ and $Q\,dK^{\top}$ are merged into ONE dot of inner width $2 d_h = 128$ by interleaving: `tl.join(dq, q)` against `tl.join(k, dk)` sums $dq_i k_i + q_i dk_i$ over the joined axis, so the kernel issues 5 dots per tile instead of 6 with identical FLOPs (floating point operations)

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

- Flash kernel implementation: exp2-based online softmax, arbitrary input strides (q, k, v, dq, dk, dv are strided views of the single stacked QKV GEMM output, no transpose copies), autotuned over tile configs {(64,64), (128,64), (64,128), (128,128)} x warps x stages with an EVEN_S specialization that drops all bounds masks when S is a multiple of 128; autotune selects BLOCK_M = 64, BLOCK_N = 64, num_warps = 4, num_stages = 3 at S = 4096

## Correctness

- Reference: pure PyTorch block under `torch.func.jvp`, fp32, math SDPA (scaled dot-product attention) backend forced via `sdpa_kernel([SDPBackend.MATH])`, on the same bf16-representable weights and inputs; threshold: relative Frobenius error $\le 3 \times 10^{-2}$ on y and dy, bf16 run

| Shape (B, S, D, H) | y rel Fro | y max abs | dy rel Fro | dy max abs |
|---|---|---|---|---|
| (2, 512, 768, 12) | 1.010e-02 | 5.287e-02 | 1.421e-02 | 7.072e-02 |
| (1, 1000, 768, 12) | 1.011e-02 | 5.052e-02 | 1.422e-02 | 7.233e-02 |
| (2, 4096, 512, 8) | 5.831e-03 | 3.483e-02 | 8.032e-03 | 4.462e-02 |

- The error is dominated by fp8 quantization of the four linears (the pure-bf16 predecessor measured about 2.5e-03); it sits 2x-5x under the 3e-02 threshold; (1, 1000, ...) exercises a sequence length that is not a multiple of the tile size; `TritonBlockJVP` module output is bit-exact versus the functional `triton_block_jvp` path (identical deterministic weight quantization); all 4 pytest cases pass

## Benchmark

- bf16 io, CUDA events, 30 iterations after 5 warmup, versus `torch.func.jvp` over the eager PyTorch block (math SDPA is the only jvp-capable PyTorch attention path) and versus `torch.compile` of the same jvp closure

| Shape (B, S, D, H) | Triton (ms) | Eager jvp (ms) | Speedup | Compiled jvp (ms) | Speedup |
|---|---|---|---|---|---|
| (2, 1024, 768, 12) | 0.501 | 6.215 | 12.4x | 2.990 | 6.0x |
| (2, 4096, 768, 12) | 3.575 | 81.684 | 22.9x | 33.191 | 9.3x |
| (1, 8192, 768, 12) | 5.810 | OOM | n/a | 61.445 | 10.6x |

- (1, 8192, 768, 12) eager baseline: CUDA out of memory on the 16 GB card (math-SDPA jvp materializes multiple 12 x 8192 x 8192 score tensors); the compiled baseline at 8192 succeeded at 61.445 ms in one run and hit OutOfMemoryError in another depending on allocator state, the Triton path runs the shape in under 3 GiB either way

- No CUDA graph capture is used anywhere; all numbers are plain eager launches for both the Triton path and the baselines

## Optimization from 13.0x to 22.9x at (2, 4096, 768, 12)

- Per-stage profile before → after (CUDA events, 50 iterations, isolated stage replays on cached buffers; before = bf16 cuBLAS GEMMs + bf16/fp32-accumulate flash kernel at 6.27 ms total, after = fp8 GEMMs + fp16-accumulate flash kernel at 3.58 ms total):

| Stage | Before (ms) | After (ms) |
|---|---|---|
| RMSNorm JVP x2 | 0.046 | 0.028 |
| QKV GEMM | 0.652 | 0.283 |
| flash-attention JVP | 3.302 | 2.232 |
| out GEMM | 0.205 | 0.085 |
| residual adds | 0.033 | 0.025 |
| gate/up GEMM | 1.141 | 0.489 |
| SwiGLU JVP | 0.255 | 0.202 |
| down GEMM | 0.542 | 0.188 |

- Measured tensor-core ceilings on this card (4096^3 Triton GEMM): bf16 with fp32 accumulate 91.8 TFLOPS, fp16 with fp16 accumulate 162.3 TFLOPS, fp8e4m3 with fp32 accumulate 151.2 TFLOPS; the old flash kernel ran at 94 TFLOPS (already at the bf16/fp32-acc wall), the new fp16-accumulate kernel runs the same 309 GFLOP in 2.23 ms = 138 TFLOPS (85 percent of the fp16-acc wall)

- fp8 `torch._scaled_mm` per-tensor scaling works on sm_120 in torch 2.11 and runs the gate/up GEMM 2.5x faster than bf16 cuBLAS (1.12 → 0.45 ms); rowwise scaling also works but is slower (0.60 ms), so per-tensor is used

- Rejected after measurement: merging $T V + \tilde{P}\, dV$ into one dot $[T | \tilde{P}]\,[V; dV]$ of inner width $2 B_N$ — the row interleave of $[V; dV]$ needs a `tl.permute` register-layout conversion and the kernel slows from 2.16 to 3.01 ms; the q/k-side join for $dS$ has no such permute (join along the head axis matches the dot operand layout) and is kept

- Rejected as unnecessary: CUDA graph capture — the stage sum (3.48 ms) is within 0.1 ms of the end-to-end time (3.58 ms), so launch overhead is already negligible at this sequence length

- Remaining wall: the flash JVP kernel holds 62 percent of the runtime at 138 TFLOPS with fp16 tile accumulation; the next factor would need fp8 score/value dots (151 TFLOPS ceiling gives at most 9 percent) or smaller-precision softmax, both of which spend accuracy for little return, and the fp8 GEMMs already run at 0.19-0.49 ms against a 210 TFLOPS effective rate

- Repro: `/opt/miniconda3/bin/python3 tests/test_triton_block_jvp.py` (correctness then benchmark), or `python3 -m pytest tests/test_triton_block_jvp.py` for correctness only
