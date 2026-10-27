# tests/unit_tests/distributed/test_dp_overlap_grad_reduce.py
import contextlib

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import TestModel, Utils


def _get_model_and_buffers(overlap_grad_reduce, use_distributed_optimizer):
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        use_distributed_optimizer=use_distributed_optimizer,
        overlap_grad_reduce=overlap_grad_reduce,
        bucket_size=None,  # single bucket -> focus on the overlap scheduling contract
        average_in_collective=False,
    )
    model = TestModel(
        input_dim=100, output_dim=100, num_layers=10, bias=True, shared_embedding=False
    ).bfloat16()
    config = TransformerConfig(num_attention_heads=1, num_layers=1, hidden_size=100)
    model = DistributedDataParallel(config=config, ddp_config=ddp_config, module=model)
    return model, model.buffers[0], model.bucket_groups

@pytest.mark.sanity
@pytest.mark.regression
@pytest.mark.internal
@pytest.mark.parametrize("use_distributed_optimizer", [False, True])
@pytest.mark.parametrize("overlap_grad_reduce", [False, True])
def test_dp_overlap_grad_reduce(overlap_grad_reduce, use_distributed_optimizer):
    """Feature: DP backward / grad-reduce overlap (overlap_grad_reduce).

    Two ranks DP2: with overlap on, finish_grad_sync must raise before the bucket is full
    (async collective not yet dispatched) and consume the handle once all params are registered;
    with overlap off, finish runs synchronously without raising.
    """
    Utils.initialize_model_parallel(1, 1)  # TP1 PP1 -> DP2 under torchrun nproc=2

    model, param_and_grad_buffer, bucket_groups = _get_model_and_buffers(
        overlap_grad_reduce=overlap_grad_reduce,
        use_distributed_optimizer=use_distributed_optimizer,
    )
    param_to_bucket_group = {p: bg for bg in bucket_groups for p in bg.params}
    param_and_grad_buffer.grad_data.data.fill_(1.0)

    params = list(model.parameters())
    # overlap=False -> register_grad_ready itself asserts overlap==True.
    register_ctx = contextlib.nullcontext() if overlap_grad_reduce else pytest.raises(AssertionError)
    for param in params:
        with register_ctx:
            param_to_bucket_group[param].register_grad_ready(param)

    if not overlap_grad_reduce:
        for bg in bucket_groups:
            bg.finish_grad_sync()
            assert bg.grad_reduce_handle is None
        Utils.destroy_model_parallel()
        return

    for bg in bucket_groups:
        bg.reset()
    for i, param in enumerate(params):
        param_to_bucket_group[param].register_grad_ready(param)
        if i < len(params) - 1:
            with pytest.raises(AssertionError):
                param_to_bucket_group[param].finish_grad_sync()
    for bg in bucket_groups:
        bg.finish_grad_sync()
        assert bg.grad_reduce_handle is None, "grad_reduce_handle not consumed"

    Utils.destroy_model_parallel()
