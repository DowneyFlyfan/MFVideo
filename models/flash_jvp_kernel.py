# Fused FlashAttention forward + JVP (forward-mode tangent) kernel for SM120
# (RTX 5070 Ti, compute capability 12.0), written in CuTeDSL (nvidia-cutlass-dsl).
#
# Computes, for Q, K, V and tangents dQ, dK, dV (all (B, S, H, D), bf16/fp16):
#   primal:  O  = softmax(scale * Q K^T) V
#   tangent: dO = P (dS - rowsum(P dS)) V + P dV        (P = normalized probs)
# using a streaming online-softmax formulation (see math below), fused into a
# single kernel with 6 GEMMs per (m, n) tile:
#   S = Q K^T, dS_raw = dQ K^T + Q dK^T,  P~ V,  T V,  P~ dV
# where P~ = exp(scale*(S - m_row)) (unnormalized), T = P~ * (scale * dS_raw).
# Per m-row streaming state: running max m_row, running sum l = rowsum(P~),
# mu = rowsum(T). Final:
#   O  = acc_O / l
#   dO = acc_dO / l - (mu / l) * (acc_O / l)
# Treating the running max as a constant shift is exact for the JVP because the
# shift cancels between numerator and denominator.
#
# Structure adapted from flash_attn.cute.flash_fwd (FlashAttentionForwardSm80)
# and flash_fwd_sm120.

import math
import operator
from types import SimpleNamespace
from typing import Type, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import warp
from cutlass.base_dsl.arch import Arch

from quack import layout_utils

from flash_attn.cute import ampere_helpers as sm80_utils
from flash_attn.cute import utils
from flash_attn.cute.cute_dsl_utils import (
    assume_tensor_aligned,
    to_cute_tensor,
    torch2cute_dtype_map,
)
from flash_attn.cute.flash_fwd import FlashAttentionForwardSm80
from flash_attn.cute.softmax import Softmax
from flash_attn.cute.seqlen_info import SeqlenInfoQK

LOG2_E = math.log2(math.e)


class FlashAttentionJvpSm120(FlashAttentionForwardSm80):
    """Fused non-causal FlashAttention forward + JVP for SM120.

    Fixed configuration: seqlen_q == seqlen_k, no varlen, no GQA, no dropout,
    no local/causal masking, no softcap, num_stages=1, Q_in_regs=False.
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        tile_m: int = 64,
        tile_n: int = 64,
        num_threads: int = 128,
    ):
        super().__init__(
            dtype,
            head_dim,
            head_dim_v=None,
            qhead_per_kvhead=1,
            is_causal=False,
            is_local=False,
            pack_gqa=False,
            tile_m=tile_m,
            tile_n=tile_n,
            num_stages=1,
            num_threads=num_threads,
            Q_in_regs=False,
        )
        # Force SM80 code paths while the DSL targets the resident SM120 GPU
        # (SM120 uses the same mma.sync.m16n8k16 instructions).
        self.arch = Arch.sm_80
        # 6 smem tiles (Q, K, V, dQ, dK, dV); SM120 has 99 KB smem.
        smem_usage = 2 * (
            tile_m * self.tile_hdim
            + tile_n * self.tile_hdim * self.num_stages
            + tile_n * self.tile_hdimv * self.num_stages
        ) * (dtype.width // 8)
        assert smem_usage <= 99 * 1024, f"smem usage {smem_usage} exceeds 99KB"

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(layout)], 1024]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout)
        ]

        @cute.struct
        class SharedStorageJVP:
            sQ: sQ_struct
            sK: sK_struct
            sV: sV_struct
            sdQ: sQ_struct
            sdK: sK_struct
            sdV: sV_struct

        return SharedStorageJVP

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdQ: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        softmax_scale: Float32,
        stream: cuda.CUstream = None,
    ):
        """Q/K/V/dQ/dK/dV/O/dO have layout (batch, seqlen, num_head, head_dim),
        same dtype. mLSE is (batch, num_head, seqlen) fp32 (FA4 stock layout);
        values are natural-log logsumexp of scale * Q K^T rows, exactly as
        produced by the stock forward and consumed by the FA4 backward."""
        all_tensors = (mQ, mK, mV, mdQ, mdK, mdV, mO, mdO)
        for t in all_tensors:
            assert t.element_type == self.dtype
        assert mLSE.element_type == Float32

        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_pv.size
        self.num_producer_threads = self.num_threads
        self.num_Q_load_threads = self.num_threads
        self.num_epilogue_threads = self.num_threads
        self.use_tma_O = False
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()

        mQ, mK, mV, mdQ, mdK, mdV, mO, mdO = [
            assume_tensor_aligned(t) for t in all_tensors
        ]
        # (B, S, H, D) -> (S, D, H, B)
        mQ, mK, mV, mdQ, mdK, mdV, mO, mdO = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=[1, 3, 2, 0]))
            for t in (mQ, mK, mV, mdQ, mdK, mdV, mO, mdO)
        ]
        # (B, H, S) -> (S, H, B), matching flash_fwd's LSE_layout_transpose.
        mLSE = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=[2, 1, 0]))

        softmax_scale_log2 = softmax_scale * LOG2_E
        grid_dim = (
            cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]),
        )
        self.kernel(
            mQ,
            mK,
            mV,
            mdQ,
            mdK,
            mdV,
            mO,
            mdO,
            mLSE,
            softmax_scale_log2,
            softmax_scale,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_K,
            self.gmem_tiled_copy_V,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdQ: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, head_idx, batch_idx = cute.arch.block_idx()

        seqlen = SeqlenInfoQK.create(
            batch_idx=batch_idx,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        n_block_max = cute.ceil_div(seqlen.seqlen_k, self.tile_n)
        n_block = n_block_max - 1

        # ///////////////////////////////////////////////////////////////////////////
        # Global tiles for this thread block
        # ///////////////////////////////////////////////////////////////////////////
        blkQ_shape = (self.tile_m, self.tile_hdim)
        blkK_shape = (self.tile_n, self.tile_hdim)
        blkV_shape = (self.tile_n, self.tile_hdimv)
        mQ_cur = mQ[None, None, head_idx, batch_idx]
        mdQ_cur = mdQ[None, None, head_idx, batch_idx]
        mK_cur = mK[None, None, head_idx, batch_idx]
        mdK_cur = mdK[None, None, head_idx, batch_idx]
        mV_cur = mV[None, None, head_idx, batch_idx]
        mdV_cur = mdV[None, None, head_idx, batch_idx]
        gQ = cute.local_tile(mQ_cur, blkQ_shape, (m_block, 0))
        gdQ = cute.local_tile(mdQ_cur, blkQ_shape, (m_block, 0))
        gK = cute.local_tile(mK_cur, blkK_shape, (None, 0))
        gdK = cute.local_tile(mdK_cur, blkK_shape, (None, 0))
        gV = cute.local_tile(mV_cur, blkV_shape, (None, 0))
        gdV = cute.local_tile(mdV_cur, blkV_shape, (None, 0))

        # ///////////////////////////////////////////////////////////////////////////
        # Shared memory
        # ///////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sK_layout)
        sV = storage.sV.get_tensor(sV_layout)
        sdQ = storage.sdQ.get_tensor(sQ_layout)
        sdK = storage.sdK.get_tensor(sK_layout)
        sdV = storage.sdV.get_tensor(sV_layout)
        sVt = layout_utils.transpose_view(sV)
        sdVt = layout_utils.transpose_view(sdV)

        gmem_thr_copy_K = gmem_tiled_copy_K.get_slice(tidx)
        gmem_thr_copy_V = gmem_tiled_copy_V.get_slice(tidx)
        tKsK, tKgK = gmem_thr_copy_K.partition_D(sK), gmem_thr_copy_K.partition_S(gK)
        tdKsdK, tdKgdK = gmem_thr_copy_K.partition_D(sdK), gmem_thr_copy_K.partition_S(gdK)
        tVsV, tVgV = gmem_thr_copy_V.partition_D(sV), gmem_thr_copy_V.partition_S(gV)
        tdVsdV, tdVgdV = gmem_thr_copy_V.partition_D(sdV), gmem_thr_copy_V.partition_S(gdV)

        # ///////////////////////////////////////////////////////////////////////////
        # MMA partitions and accumulators
        # ///////////////////////////////////////////////////////////////////////////
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        tSrdQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sdQ))
        tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK[None, None, 0]))
        tSrdK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sdK[None, None, 0]))
        tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt[None, None, 0]))
        tOrdVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sdVt[None, None, 0]))
        acc_shape_O = thr_mma_pv.partition_shape_C((self.tile_m, self.tile_hdimv))
        acc_O = cute.make_rmem_tensor(acc_shape_O, Float32)
        acc_dO = cute.make_rmem_tensor(acc_shape_O, Float32)
        acc_O.fill(0.0)
        acc_dO.fill(0.0)

        # ///////////////////////////////////////////////////////////////////////////
        # Smem -> rmem copy partitions (ldmatrix)
        # ///////////////////////////////////////////////////////////////////////////
        smem_copy_atom_QK = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
        )
        smem_thr_copy_Q = utils.make_tiled_copy_A(smem_copy_atom_QK, tiled_mma_qk).get_slice(tidx)
        smem_thr_copy_K = utils.make_tiled_copy_B(smem_copy_atom_QK, tiled_mma_qk).get_slice(tidx)
        smem_thr_copy_V = utils.make_tiled_copy_B(smem_copy_atom_V, tiled_mma_pv).get_slice(tidx)
        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSsdQ = smem_thr_copy_Q.partition_S(sdQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tSsdK = smem_thr_copy_K.partition_S(sdK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)
        tOsdVt = smem_thr_copy_V.partition_S(sdVt)

        # ///////////////////////////////////////////////////////////////////////////
        # Predicates for K/V loads (head_dim direction)
        # ///////////////////////////////////////////////////////////////////////////
        cK = cute.make_identity_tensor((self.tile_n, self.tile_hdim))
        tKcK = gmem_thr_copy_K.partition_S(cK)
        t0KcK = gmem_thr_copy_K.get_slice(0).partition_S(cK)
        if const_expr(self.tile_hdim == self.tile_hdimv):
            tVcV, t0VcV = tKcK, t0KcK
        else:
            cV = cute.make_identity_tensor((self.tile_n, self.tile_hdimv))
            tVcV = gmem_thr_copy_V.partition_S(cV)
            t0VcV = gmem_thr_copy_V.get_slice(0).partition_S(cV)
        tKpK = utils.predicate_k(tKcK, limit=mK.shape[1])
        tVpV = tKpK if const_expr(self.same_hdim_kv) else utils.predicate_k(tVcV, limit=mV.shape[1])

        # ///////////////////////////////////////////////////////////////////////////
        # Softmax state: running max/sum (l) via Softmax; mu = rowsum(T) partials
        # ///////////////////////////////////////////////////////////////////////////
        num_rows = acc_O.shape[0][0] * acc_O.shape[1]
        softmax = Softmax.create(softmax_scale_log2, num_rows=num_rows)
        softmax.reset()
        mu = cute.make_rmem_tensor(num_rows, Float32)
        mu.fill(0.0)

        load_K = partial(
            self.load_K, gmem_tiled_copy_K, tKgK, tKsK, tKcK, t0KcK, tKpK,
            seqlen=seqlen.seqlen_k,
        )
        load_dK = partial(
            self.load_K, gmem_tiled_copy_K, tdKgdK, tdKsdK, tKcK, t0KcK, tKpK,
            seqlen=seqlen.seqlen_k,
        )
        load_V = partial(
            self.load_V, gmem_tiled_copy_V, tVgV, tVsV, tVcV, t0VcV, tVpV,
            seqlen=seqlen.seqlen_k,
        )
        load_dV = partial(
            self.load_V, gmem_tiled_copy_V, tdVgdV, tdVsdV, tVcV, t0VcV, tVpV,
            seqlen=seqlen.seqlen_k,
        )

        mma_params = SimpleNamespace(
            thr_mma_qk=thr_mma_qk,
            thr_mma_pv=thr_mma_pv,
            tSrQ=tSrQ,
            tSrdQ=tSrdQ,
            tSrK=tSrK,
            tSrdK=tSrdK,
            tOrVt=tOrVt,
            tOrdVt=tOrdVt,
            acc_O=acc_O,
            acc_dO=acc_dO,
        )
        smem_copy_params = SimpleNamespace(
            smem_thr_copy_Q=smem_thr_copy_Q,
            smem_thr_copy_K=smem_thr_copy_K,
            smem_thr_copy_V=smem_thr_copy_V,
            tSsQ=tSsQ,
            tSsdQ=tSsdQ,
            tSsK=tSsK,
            tSsdK=tSsdK,
            tOsVt=tOsVt,
            tOsdVt=tOsdVt,
        )
        compute_one_n_block = partial(
            self.compute_one_n_block_jvp,
            mma_params=mma_params,
            smem_copy_params=smem_copy_params,
            softmax=softmax,
            mu=mu,
            softmax_scale=softmax_scale,
            load_K=load_K,
            load_dK=load_dK,
            load_V=load_V,
            load_dV=load_dV,
            seqlen=seqlen,
        )

        # ///////////////////////////////////////////////////////////////////////////
        # Prologue: async loads of Q/dQ then K/dK for the last n-block
        # ///////////////////////////////////////////////////////////////////////////
        gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
        self.load_Q(gmem_thr_copy_Q, gQ, sQ, m_block, seqlen=seqlen.seqlen_q, headdim=mQ.shape[1])
        self.load_Q(gmem_thr_copy_Q, gdQ, sdQ, m_block, seqlen=seqlen.seqlen_q, headdim=mQ.shape[1])
        cute.arch.cp_async_commit_group()
        load_K(n_block, smem_pipe_write=0, need_predicates=True)
        load_dK(n_block, smem_pipe_write=0, need_predicates=True)
        cute.arch.cp_async_commit_group()

        # ///////////////////////////////////////////////////////////////////////////
        # Mainloop (n-blocks processed from last to first)
        # ///////////////////////////////////////////////////////////////////////////
        compute_one_n_block(n_block, is_first_n_block=True, mask_seqlen=True)
        for n_tile in cutlass.range(n_block, unroll=1):
            compute_one_n_block(n_block - n_tile - 1, is_first_n_block=False, mask_seqlen=False)

        # ///////////////////////////////////////////////////////////////////////////
        # Finalize: quad-allreduce l and mu, normalize, tangent correction
        # ///////////////////////////////////////////////////////////////////////////
        mu.store(utils.warp_reduce(mu.load(), operator.add, width=4))
        row_scale = softmax.finalize()  # row_scale = 1 / l
        softmax.rescale_O(acc_O, row_scale)   # acc_O  <- O = acc_O / l
        softmax.rescale_O(acc_dO, row_scale)  # acc_dO <- acc_dO / l
        acc_O_mn = layout_utils.reshape_acc_to_mn(acc_O)
        acc_dO_mn = layout_utils.reshape_acc_to_mn(acc_dO)
        for r in cutlass.range(num_rows, unroll_full=True):
            acc_dO_mn[r, None].store(
                acc_dO_mn[r, None].load() - (mu[r] * row_scale[r]) * acc_O_mn[r, None].load()
            )

        # ///////////////////////////////////////////////////////////////////////////
        # Epilogue: write O (smem reuse of sQ), then dO (smem reuse of sdQ)
        # ///////////////////////////////////////////////////////////////////////////
        # softmax.finalize() left row_sum holding the natural-log logsumexp:
        # row_sum = row_max * scale_log2 * ln2 + ln(l) = scale * row_max + ln(l),
        # exactly the LSE the stock FA4 forward stores and the backward consumes.
        sO = cute.make_tensor(sQ.iterator, sO_layout)
        self.epilogue(
            acc_O, softmax.row_sum, mO, mLSE, sO, seqlen,
            gmem_tiled_copy_O, None, tiled_mma_pv, tidx, m_block, head_idx, batch_idx,
        )
        sdO = cute.make_tensor(sdQ.iterator, sO_layout)
        self.epilogue(
            acc_dO, softmax.row_sum, mdO, None, sdO, seqlen,
            gmem_tiled_copy_O, None, tiled_mma_pv, tidx, m_block, head_idx, batch_idx,
        )

    @cute.jit
    def compute_one_n_block_jvp(
        self,
        n_block: Int32,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        mu: cute.Tensor,
        softmax_scale: Float32,
        load_K,
        load_dK,
        load_V,
        load_dV,
        seqlen: SeqlenInfoQK,
        mask_seqlen: cutlass.Constexpr[bool],
        is_first_n_block: cutlass.Constexpr[bool] = False,
    ):
        """Process one n-block: 3 QK-type GEMMs, online softmax + tangent
        elementwise math, 3 PV-type GEMMs."""

        def sync():
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

        acc_shape_S = mma_params.thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
        acc_S = cute.make_rmem_tensor(acc_shape_S, Float32)
        acc_dS = cute.make_rmem_tensor(acc_shape_S, Float32)
        acc_S.fill(0.0)
        acc_dS.fill(0.0)

        # Wait for Q/dQ (first iteration) and K/dK tiles.
        sync()

        # Issue async loads of V/dV for this n-block (needs predicates only on the
        # first processed block, which is the seqlen tail block).
        load_V(n_block, smem_pipe_write=0, need_predicates=is_first_n_block)
        load_dV(n_block, smem_pipe_write=0, need_predicates=is_first_n_block)
        cute.arch.cp_async_commit_group()

        # GEMM 1: acc_S = Q K^T (loads Q and K fragments from smem)
        sm80_utils.gemm(
            mma_params.thr_mma_qk, acc_S, mma_params.tSrQ, mma_params.tSrK,
            smem_copy_params.tSsQ, smem_copy_params.tSsK[None, None, None, 0],
            smem_copy_params.smem_thr_copy_Q, smem_copy_params.smem_thr_copy_K,
        )
        # GEMM 2: acc_dS += dQ K^T (K fragment already in registers)
        sm80_utils.gemm(
            mma_params.thr_mma_qk, acc_dS, mma_params.tSrdQ, mma_params.tSrK,
            smem_copy_params.tSsdQ, smem_copy_params.tSsK[None, None, None, 0],
            smem_copy_params.smem_thr_copy_Q, smem_copy_params.smem_thr_copy_K,
            B_in_regs=True,
        )
        # GEMM 3: acc_dS += Q dK^T (Q fragment already in registers)
        sm80_utils.gemm(
            mma_params.thr_mma_qk, acc_dS, mma_params.tSrQ, mma_params.tSrdK,
            smem_copy_params.tSsQ, smem_copy_params.tSsdK[None, None, None, 0],
            smem_copy_params.smem_thr_copy_Q, smem_copy_params.smem_thr_copy_K,
            A_in_regs=True,
        )

        # Wait for V/dV; then start loading next K/dK (overwrites sK/sdK, whose
        # reads finished above; the barrier in sync() orders all threads).
        sync()
        if n_block - 1 >= 0:
            load_K(n_block - 1, smem_pipe_write=0, need_predicates=False)
            load_dK(n_block - 1, smem_pipe_write=0, need_predicates=False)
        cute.arch.cp_async_commit_group()

        # Seqlen masking on the tail block: S -> -inf (P~ = 0) and dS -> 0 so the
        # tangent path sees no garbage from unloaded K/dK smem rows.
        if const_expr(mask_seqlen):
            self.apply_seqlen_mask_jvp(
                acc_S, acc_dS, mma_params.thr_mma_qk, n_block, seqlen.seqlen_k
            )

        # Online softmax: acc_S -> P~ = exp(scale*(S - m_row)); returns rescale
        # factor for the running accumulators when the max changed.
        row_scale = softmax.online_softmax(acc_S, is_first=is_first_n_block, check_inf=True)
        softmax.rescale_O(mma_params.acc_O, row_scale)
        softmax.rescale_O(mma_params.acc_dO, row_scale)
        for r in cutlass.range(cute.size(mu), unroll_full=True):
            mu[r] *= row_scale[r]

        # T = P~ * (scale * dS_raw) in fp32; mu += rowsum(T).
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        acc_dS_mn = layout_utils.reshape_acc_to_mn(acc_dS)
        for r in cutlass.range(cute.size(mu), unroll_full=True):
            t_row = acc_S_mn[r, None].load() * (acc_dS_mn[r, None].load() * softmax_scale)
            acc_dS_mn[r, None].store(t_row)
            mu[r] = utils.fadd_reduce(t_row, init_val=mu[r], arch=80)

        # Convert P~ and T to input dtype as mma A-operands.
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        rT = cute.make_fragment_like(acc_dS, self.dtype)
        rT.store(acc_dS.load().to(self.dtype))
        tOrP = layout_utils.reshape_acc_to_frgA(rP)
        tOrT = layout_utils.reshape_acc_to_frgA(rT)

        # GEMM 4: acc_O += P~ V (loads V fragment from smem)
        sm80_utils.gemm_rs(
            mma_params.thr_mma_pv, mma_params.acc_O, tOrP, mma_params.tOrVt,
            smem_copy_params.tOsVt[None, None, None, 0],
            smem_copy_params.smem_thr_copy_V,
        )
        # GEMM 5: acc_dO += T V (V fragment already in registers)
        for k in cutlass.range_constexpr(cute.size(tOrT.shape[2])):
            cute.gemm(
                mma_params.thr_mma_pv, mma_params.acc_dO,
                tOrT[None, None, k], mma_params.tOrVt[None, None, k],
                mma_params.acc_dO,
            )
        # GEMM 6: acc_dO += P~ dV (loads dV fragment from smem)
        sm80_utils.gemm_rs(
            mma_params.thr_mma_pv, mma_params.acc_dO, tOrP, mma_params.tOrdVt,
            smem_copy_params.tOsdVt[None, None, None, 0],
            smem_copy_params.smem_thr_copy_V,
        )

    @cute.jit
    def apply_seqlen_mask_jvp(
        self,
        acc_S: cute.Tensor,
        acc_dS: cute.Tensor,
        thr_mma,
        n_block: Int32,
        seqlen_k: Int32,
    ):
        """Mask out-of-range columns: acc_S -> -inf, acc_dS -> 0."""
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        acc_dS_mn = layout_utils.reshape_acc_to_mn(acc_dS)
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cS))
        t0ScS_mn = layout_utils.reshape_acc_to_mn(thr_mma.get_slice(0).partition_C(cS))
        thr_col_offset = tScS_mn[0][1]
        seqlenk_col_limit = seqlen_k - n_block * self.tile_n - thr_col_offset
        for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
            oob = t0ScS_mn[0, c][1] >= seqlenk_col_limit
            for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                acc_S_mn[r, c] = -Float32.inf if oob else acc_S_mn[r, c]
                acc_dS_mn[r, c] = 0.0 if oob else acc_dS_mn[r, c]


# /////////////////////////////////////////////////////////////////////////////
# Python wrapper
# /////////////////////////////////////////////////////////////////////////////

_jvp_compile_cache = {}


def _check_buffer(buf, name, shape, dtype, device):
    assert buf.shape == shape, f"{name}: shape {tuple(buf.shape)} != {tuple(shape)}"
    assert buf.dtype == dtype, f"{name}: dtype {buf.dtype} != {dtype}"
    assert buf.is_cuda and buf.device == device, f"{name}: wrong device"
    assert buf.is_contiguous(), f"{name} must be contiguous"
    return buf


def flash_attn_jvp_func(
    q, k, v, tq, tk, tv,
    softmax_scale: Optional[float] = None,
    out=None, t_out=None, lse_out=None,
):
    """Fused FlashAttention forward + JVP with logsumexp output.

    Args:
        q, k, v: (batch, seqlen, num_head, head_dim) bf16/fp16 (fp32 is cast to bf16)
        tq, tk, tv: tangents of q, k, v, same shape/dtype
        softmax_scale: defaults to 1/sqrt(head_dim)
        out, t_out: optional preallocated output buffers, same shape/dtype as the
            (possibly bf16-cast) q, contiguous; written in place when given.
        lse_out: optional preallocated (batch, num_head, seqlen) fp32 buffer for
            the natural-log logsumexp (FA4 stock layout), written in place.

    Returns:
        (o, t_o, lse): attention output, its tangent, and the logsumexp.
    """
    import torch

    assert q.dim() == 4, "expected (batch, seqlen, num_head, head_dim)"
    orig_dtype = q.dtype
    tensors = [q, k, v, tq, tk, tv]
    if orig_dtype == torch.float32:
        tensors = [t.to(torch.bfloat16) for t in tensors]
    q, k, v, tq, tk, tv = [t.contiguous() for t in tensors]
    assert q.dtype in (torch.bfloat16, torch.float16), "only bf16/fp16 supported"
    for t in (k, v, tq, tk, tv):
        assert t.shape == q.shape and t.dtype == q.dtype and t.is_cuda
    batch, seqlen, num_head, head_dim = q.shape
    assert k.shape[1] == seqlen, "seqlen_q must equal seqlen_k"
    assert head_dim % 8 == 0, "head_dim must be a multiple of 8"

    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5
    lse_shape = (batch, num_head, seqlen)
    if out is None:
        out = torch.empty_like(q)
    else:
        _check_buffer(out, "out", q.shape, q.dtype, q.device)
    if t_out is None:
        t_out = torch.empty_like(q)
    else:
        _check_buffer(t_out, "t_out", q.shape, q.dtype, q.device)
    if lse_out is None:
        lse_out = torch.empty(lse_shape, dtype=torch.float32, device=q.device)
    else:
        _check_buffer(lse_out, "lse_out", torch.Size(lse_shape), torch.float32, q.device)
    o, t_o, lse = out, t_out, lse_out

    dtype = torch2cute_dtype_map[q.dtype]
    key = (dtype, head_dim)
    if key not in _jvp_compile_cache:
        fa_jvp = FlashAttentionJvpSm120(dtype, head_dim, tile_m=64, tile_n=64, num_threads=128)
        cute_tensors = [to_cute_tensor(t) for t in (q, k, v, tq, tk, tv, o, t_o, lse)]
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        _jvp_compile_cache[key] = cute.compile(
            fa_jvp,
            *cute_tensors,
            softmax_scale,
            fake_stream,
            options="--enable-tvm-ffi",
        )
    _jvp_compile_cache[key](
        q.detach(), k.detach(), v.detach(), tq.detach(), tk.detach(), tv.detach(),
        o, t_o, lse, softmax_scale,
    )
    if orig_dtype == torch.float32:
        o = o.to(orig_dtype)
        t_o = t_o.to(orig_dtype)
    return o, t_o, lse
