import os

import torch


def get_num_ranks_per_node(group_size: int) -> int:
    override = os.getenv('MEGATRON_NUM_RANKS_PER_NODE')
    if override:
        num_ranks_per_node = int(override)
        if num_ranks_per_node <= 0:
            raise ValueError('MEGATRON_NUM_RANKS_PER_NODE must be positive')
        return min(group_size, num_ranks_per_node)

    return min(group_size, torch.cuda.device_count())


def get_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int):
    from deep_ep import Buffer
    from megatron.core.transformer.moe import fused_a2a

    num_nvl_bytes, num_rdma_bytes = 0, 0
    group_size = group.size()
    num_ranks_per_node = get_num_ranks_per_node(group_size)
    for config in (
        Buffer.get_dispatch_config(group_size),
        Buffer.get_combine_config(group_size),
    ):
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group_size, num_ranks_per_node),
            num_nvl_bytes,
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group_size, num_ranks_per_node),
            num_rdma_bytes,
        )

    if (
        fused_a2a._buffer is None
        or fused_a2a._buffer.group != group
        or fused_a2a._buffer.num_nvl_bytes < num_nvl_bytes
        or fused_a2a._buffer.num_rdma_bytes < num_rdma_bytes
    ):
        fused_a2a._buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return fused_a2a._buffer
