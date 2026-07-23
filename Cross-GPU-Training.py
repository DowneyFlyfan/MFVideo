import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torchao.float8 import convert_to_float8_training  # fp8 linears (H100 TensorCores)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % 8)

    # -- 2D mesh: ("replicate"=2 nodes) x ("shard"=8 GPUs/node) --------------
    mesh = init_device_mesh("cuda", (2, 8), mesh_dim_names=("replicate", "shard"))

    model = build_transformer().cuda()  # meta-init for huge models

    # -- fp8 compute for all big linears (rowwise scaling, ~1.3-1.5x speedup)
    convert_to_float8_training(model, module_filter_fn=lambda m, n: "lm_head" not in n)

    # -- selective activation checkpointing: recompute cheap ops, save matmuls
    for i, block in enumerate(model.blocks):
        model.blocks[i] = checkpoint_wrapper(block)

    # -- FSDP2 (per-parameter DTensor sharding), bf16 compute / fp32 reduce --
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for block in model.blocks:  # per-block: overlaps comm/compute
        fully_shard(block, mesh=mesh, mp_policy=mp)
    fully_shard(model, mesh=mesh, mp_policy=mp)

    # -- compile: kernel fusion + max-autotune GEMM selection ----------------
    model = torch.compile(model)

    # -- fused optimizer, fp32 sharded states --------------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)

    for step, batch in enumerate(loader):  # pre-packed tokens, pinned mem
        x = batch.cuda(non_blocking=True)
        # SDPA dispatches to FlashAttention-3-class kernel (TMA/WGMMA) on H100
        loss = model(x).loss / GRAD_ACCUM
        loss.backward()  # reduce-scatter overlapped here
        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # DTensor-aware
            opt.step()
            opt.zero_grad(set_to_none=True)
