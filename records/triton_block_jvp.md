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

## fp32 io Mode

- API (application programming interface): `triton_block_jvp(x, dx, params, eps, variant)` dispatches on the input dtype — bf16/fp16 inputs take the original path unchanged, fp32 inputs (with fp32-stored params) return fp32 (y, dy) with the internals selected by `variant`: \textbf{fast} (default), \textbf{hp} (high precision, see the dedicated section below), or \textbf{tf32}; `TritonBlockJVP(..., dtype=torch.float32, variant=...)` exposes the same choice, and all pre-existing bf16 call sites keep working without modification

- \textbf{fast} variant internals (io is fp32 but compute is NOT): parameters are stored fp32 and amax-quantized once to fp8 (float8_e4m3fn) for the four linears; RMSNorm1/RMSNorm2 read fp32 and write fp8 directly (the downcast is fused into the existing kernels via a CLAMP constexpr, no separate cast passes); the QKV `torch._scaled_mm` emits fp16 and attention runs the fp16-accumulate flash JVP kernel unchanged; the out, gate-up, and down `torch._scaled_mm` calls emit fp32 directly (out_dtype=fp32 works on sm_120), so the residual stream, the SwiGLU input, and both final outputs are fp32 — accuracy is therefore fp8/fp16-limited, identical in kind to the bf16 path

- \textbf{tf32} variant internals: every buffer is fp32; the same flash/RMSNorm/SwiGLU Triton kernels run with fp32 operands and every `tl.dot` at tf32 inner precision, and the four linears run through cuBLAS with tf32 enabled locally inside the call (`torch.backends.cuda.matmul.fp32_precision` set to "tf32" and restored in a finally block, so the caller's global setting is never changed); no fp8 or fp16 anywhere

- fp32 operands double the k/dk/v/dv tile footprint in the flash kernel, so the tf32 path uses its own autotune config space — measured against the 101376-byte shared-memory limit of this sm_120 part, only {(64, 64) stages 1, (64, 32) stages 2, (32, 32) stages 2-3} fit (every fp16-path config needs 114688 to 360448 bytes with fp32 operands); implemented as two autotuner entry points over one kernel body parameterized by FP16_MMA/CLAMP constexpr flags

## fp32 Correctness

- Reference: the same eager block under `torch.func.jvp` in \textbf{fp64} (math SDPA), which fit in memory at every correctness shape, so no fp32-reference fallback was needed; thresholds: fast variant relative Frobenius error $\le 3 \times 10^{-2}$, tf32 variant $\le 3 \times 10^{-3}$

| Shape (B, S, D, H) | Variant | y rel Fro | y max abs | dy rel Fro | dy max abs |
|---|---|---|---|---|---|
| (2, 512, 768, 12) | fast | 9.818e-03 | 4.826e-02 | 1.400e-02 | 7.302e-02 |
| (1, 1000, 768, 12) | fast | 9.822e-03 | 4.840e-02 | 1.400e-02 | 7.144e-02 |
| (2, 4096, 512, 8) | fast | 5.362e-03 | 3.101e-02 | 7.692e-03 | 4.336e-02 |
| (2, 512, 768, 12) | tf32 | 7.785e-05 | 4.050e-04 | 1.111e-04 | 5.524e-04 |
| (1, 1000, 768, 12) | tf32 | 7.735e-05 | 3.787e-04 | 1.105e-04 | 6.446e-04 |
| (2, 4096, 512, 8) | tf32 | 4.214e-05 | 2.182e-04 | 6.036e-05 | 3.297e-04 |

- The tf32 variant sits 27x under its 3e-3 threshold; the fast variant error matches the bf16-path level (fp8 weight quantization dominates); all 10 pytest cases pass (3 bf16 shapes unchanged, 6 fp32 shape-variant combinations, 1 module-versus-functional bit-exactness)

## fp32 Benchmark

- fp32 io, CUDA events, 30 iterations after 5 warmup; baselines are the eager fp32 block under `torch.func.jvp` (math SDPA) and `torch.compile` of the same closure, both run at torch DEFAULTS: `torch.backends.cuda.matmul.allow_tf32 = False` and `torch.backends.cuda.matmul.fp32_precision = "none"` — tf32 is NOT enabled for either baseline (torch.compile even warns that tf32 is available but unused)

| Shape (B, S, D, H) | fast (ms) | tf32 (ms) | Eager (ms) | Compiled (ms) | fast/eager | fast/comp | tf32/eager | tf32/comp |
|---|---|---|---|---|---|---|---|---|
| (2, 1024, 768, 12) | 0.584 | 2.057 | 8.850 | 5.846 | 15.2x | 10.0x | 4.3x | 2.8x |
| (2, 4096, 768, 12) | 4.006 | 13.795 | 87.451 | 44.432 | 21.8x | 11.1x | 6.3x | 3.2x |
| (1, 8192, 768, 12) | 6.169 | 21.480 | OOM | 72.910 | n/a | 11.8x | n/a | 3.4x |

- The fast variant clears the 20x goal at (2, 4096, 768, 12): 21.8x versus the fp32 eager baseline; the tf32 variant does NOT reach 20x (6.3x) — its flash kernel is capped by the small shared-memory-viable tiles (single-stage pipelining at (64, 64)) and the tf32 MMA rate, so the higher accuracy costs 3.4x over the fast variant

- fp32 io costs the fast path 0.43 ms over bf16 io at (2, 4096, 768, 12) (4.006 versus 3.572 ms in the same run): the extra traffic is the fp32 residual stream, the fp32 gate/up GEMM output read by SwiGLU, and the fp32 final writes

- (1, 8192, 768, 12) eager fp32 baseline: CUDA out of memory on the 16 GB card, the same failure mode as the bf16 eager baseline at that shape

## High-Precision fp32 Variant (hp): 14.2x vs the tf32-Enabled Eager Baseline

- Goal: an accuracy-preserving fp32-io path at $\ge 10\times$ the \textbf{tf32-enabled} eager baseline (the previous high-accuracy tf32 variant sat at 13.8 ms $= 5.5\times$), with relative Frobenius error $\le 3 \times 10^{-3}$ versus the fp64 `torch.func.jvp` reference

- Baseline measurement (torch 2.11.0+cu130): `torch.backends.cuda.matmul.allow_tf32 = True` (plus `torch.backends.cudnn.allow_tf32 = True`) and `torch.backends.cuda.matmul.fp32_precision = "tf32"` are the SAME flag in this torch version — setting `allow_tf32 = True` makes `fp32_precision` read "tf32", and both settings measure identically; mixing reads of the legacy and new APIs after a write raises a RuntimeError, so each was verified in a separate process

- Measured tf32-enabled eager fp32 `torch.func.jvp` baseline at (2, 4096, 768, 12): \textbf{76.35 ms} (76.311 ms via `allow_tf32=True`, 76.319 ms via `fp32_precision="tf32"`, in-suite rerun 76.353 ms; tf32-off eager is 87.5 ms — the modest gain reflects that the math-SDPA jvp is largely memory-bound and that tf32 cuBLAS only reaches 46.2 TFLOPS (tera floating point operations per second) on this card, measured on a 4096^3 matmul (fp32 without tf32: about 23 TFLOPS))

- \textbf{hp} scheme (implemented as `_block_jvp_hp`, `variant="hp"`): fp16 weights (cast once from the fp32 parameters and cached, `_fp16_weight_cache`) + fp16 activations; the four linears run as plain cuBLAS fp16 GEMMs with fp32 accumulation; attention runs the UNCHANGED fp16-accumulate flash JVP kernel (fp16 q/k/v io, fp32 softmax statistics and cross-tile accumulators); RMSNorm reads the fp32 residual stream and writes fp16 directly (cast fused in the kernel, no extra pass); the residual stream, both residual adds (`torch.add(fp16, fp32, out=fp32)` fused mixed-dtype adds), and both outputs are fp32; \textbf{no fp8 anywhere}, so the fp8-quantization error of the fast variant is eliminated and the remaining error is fp16-rounding-limited (fp16 mantissa 11 bits, unit roundoff $2^{-11} \approx 4.9 \times 10^{-4}$, versus 8 bits for bf16 and 11 bits for tf32)

- hp correctness versus the fp64 `torch.func.jvp` reference (threshold $3 \times 10^{-3}$):

| Shape (B, S, D, H) | y rel Fro | y max abs | dy rel Fro | dy max abs |
|---|---|---|---|---|
| (2, 512, 768, 12) | 9.558e-05 | 4.951e-04 | 1.355e-04 | 7.527e-04 |
| (1, 1000, 768, 12) | 9.550e-05 | 4.980e-04 | 1.355e-04 | 7.588e-04 |
| (2, 4096, 512, 8) | 5.201e-05 | 2.920e-04 | 7.418e-05 | 4.210e-04 |

- The hp error (1.4e-4) is 22x under the 3e-3 budget and statistically indistinguishable from the fp32-storage tf32 variant (1.1e-4) — fp16 storage plus fp16-per-tile accumulation costs almost nothing over tf32 dots on fp32 buffers, because both carry 11-bit mantissas through the dominant dots

- fp32 io benchmark (CUDA events, 30 iterations after 5 warmup; eager_tf32on is the eager fp32 `torch.func.jvp` baseline with `fp32_precision="tf32"` enabled during its timing only):

| Shape (B, S, D, H) | hp (ms) | eager tf32-on (ms) | hp speedup | eager tf32-off (ms) | compiled (ms) | hp/comp |
|---|---|---|---|---|---|---|
| (2, 1024, 768, 12) | 0.913 | 6.991 | 7.7x | 8.836 | 5.858 | 6.4x |
| (2, 4096, 768, 12) | 5.388 | 76.353 | \textbf{14.2x} | 87.497 | 44.543 | 8.3x |
| (1, 8192, 768, 12) | 7.537 | OOM | n/a | OOM | 73.024 | 9.7x |

- At the target shape (2, 4096, 768, 12) the hp variant reaches \textbf{14.2x} against the tf32-enabled eager baseline (goal: 10x = 7.64 ms, achieved 5.39 ms); the previous tf32 variant is kept unchanged as the fp32-storage fallback (13.79 ms = 5.5x, error 1.1e-4); the fast and bf16 paths are untouched (bf16 22.8x, fast 21.8x versus their tf32-off eager baselines, all pre-existing tests pass — 13 pytest cases total including the 3 new hp shapes)

- hp per-stage profile at (2, 4096, 768, 12) (CUDA events, 50 iterations, isolated stage replays on cached buffers; stage sum 5.39 ms matches end-to-end 5.36 ms, launch overhead negligible):

| Stage | hp (ms) | Rate |
|---|---|---|
| RMSNorm JVP x2 | 0.158 | memory-bound (fp32 read) |
| QKV GEMM (fp16) | 0.647 | 90 TFLOPS |
| flash-attention JVP (fp16 acc) | 2.139 | 138 TFLOPS |
| out GEMM (fp16) | 0.213 | 91 TFLOPS |
| residual adds (fp32) | 0.160 | memory-bound |
| gate/up GEMM (fp16) | 1.124 | 92 TFLOPS |
| SwiGLU JVP | 0.256 | memory-bound |
| down GEMM (fp16) | 0.534 | 96 TFLOPS |
| final residual add (fp32) | 0.161 | memory-bound |

- Measured and rejected: `torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True` changes neither speed (5.391 versus 5.421 ms stage sum) nor error — cuBLAS keeps fp32 accumulation for these shapes on sm_120, so the GEMMs sit at the 92-TFLOPS fp16/fp32-accumulate wall either way

- Not needed: the planned split-float emulation schemes (3xBF16 hi/lo decomposition, fp16 hi + $2^{11}$-scaled lo correction dots) — plain fp16 storage already lands 22x under the accuracy budget, so spending 2-3x the dot FLOPs on correction terms would buy accuracy that the 3e-3 gate does not require; likewise smem-capped tf32 flash tiles are strictly dominated, since tf32 MMA peaks at 46 TFLOPS on this card versus 138 TFLOPS achieved by the existing fp16-accumulate kernel

- Remaining wall for hp: flash kernel 2.14 ms at 138 TFLOPS (85 percent of the 162-TFLOPS fp16-accumulate ceiling) plus 2.52 ms of cuBLAS fp16 GEMMs at the 92-TFLOPS fp32-accumulate rate; pushing the GEMMs to fp16 accumulation via a custom Triton GEMM (162 TFLOPS ceiling) could save at most about 1.2 ms (5.4 to about 4.2 ms, 18x) and is the next lever if ever required

- Repro: `/opt/miniconda3/bin/python3 tests/test_triton_block_jvp.py` (correctness then benchmark), or `python3 -m pytest tests/test_triton_block_jvp.py` for correctness only
