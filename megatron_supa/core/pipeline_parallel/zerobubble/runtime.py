import contextlib
from typing import Callable, Iterator, List, Optional, Union

import torch
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import (
    backward_step,
    check_first_val_step,
    clear_embedding_activation_buffer,
    deallocate_output_tensor,
    finish_embedding_wgrad_compute,
    forward_step,
    get_tensor_shapes,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.cuda_graphs import create_cudagraphs
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.utils import get_model_config

from megatron_supa.core.zbpp_utils import WeightGradStore

from .scheduler.graph import FuncType
from .scheduler.passes import run_schedule_passes
from .scheduler.zb import get_zb_schedule


def get_zero_bubble_forward_backward_func():
    return forward_backward_zero_bubble


def forward_backward_zero_bubble(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
    force_all_reduce: Optional[bool] = False,
):
    if isinstance(model, list):
        assert len(model) == 1, 'zero bubble schedule does not support model chunking yet'
        model = model[0]
    if isinstance(data_iterator, list):
        assert len(data_iterator) == 1, 'zero bubble schedule does not support model chunking yet'
        data_iterator = data_iterator[0]
    if adjust_tensor_shapes_fn is not None:
        raise AssertionError('zero bubble schedule does not support adjust_tensor_shapes_fn yet')
    if forward_only:
        raise AssertionError('zero bubble schedule does not support forward_only yet')

    config = get_model_config(model)
    if config.overlap_p2p_comm:
        raise ValueError('zero bubble schedule does not support overlapping p2p communication yet')

    if p2p_communicator is None and pg_collection is None:
        p2p_communicator = P2PCommunicator(pp_group=parallel_state.get_pipeline_model_parallel_group(), config=config)
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.pp = pp_group
        pg_collection.embd = parallel_state.get_embedding_group(check_initialized=False)
        pg_collection.pos_embd = parallel_state.get_position_embedding_group(check_initialized=False)
        pg_collection.cp = cp_group
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )
        pg_collection.tp_dp_cp = parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True)
    elif p2p_communicator is not None and pg_collection is not None:
        tp_group = pg_collection.tp
        cp_group = pg_collection.cp
    else:
        raise ValueError('Provide both p2p_communicator and pg_collection, or neither')

    recv_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
    )
    send_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
    )

    if config.finalize_model_grads_func is not None:
        embedding_module = clear_embedding_activation_buffer(config, model, p2p_communicator.is_pp_last_stage)
    else:
        embedding_module = None

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    no_sync_func = config.no_sync_func or contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    schedule = run_schedule_passes(
        get_zb_schedule(
            num_microbatches,
            p2p_communicator.total_stages,
            p2p_communicator.current_stage,
        )
    )
    input_tensors = {}
    output_tensors = {}
    output_tensor_grads = {}
    input_tensor_grads = {}
    forward_data_store = []
    total_num_tokens = torch.zeros([], dtype=torch.int, device='cuda')
    last_backward_microbatch = num_microbatches - 1

    for node in schedule:
        microbatch = node.microbatch
        if node.type == FuncType.RECV_FORWARD:
            input_tensors[microbatch] = p2p_communicator.recv_forward(
                recv_tensor_shapes, p2p_communicator.is_pp_first_stage
            )
        elif node.type == FuncType.F:
            output_tensor, num_tokens = forward_step(
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                input_tensors[microbatch],
                forward_data_store,
                config,
                cp_group.size(),
                collect_non_loss_data=collect_non_loss_data,
                checkpoint_activations_microbatch=None,
                is_first_microbatch=check_first_val_step(first_val_step, forward_only, microbatch == 0),
                current_microbatch=microbatch,
                is_last_stage=p2p_communicator.is_pp_last_stage,
            )
            output_tensors[microbatch] = output_tensor
            total_num_tokens += num_tokens
        elif node.type == FuncType.SEND_FORWARD:
            p2p_communicator.send_forward(output_tensors[microbatch], p2p_communicator.is_pp_last_stage)
            deallocate_output_tensor(output_tensors[microbatch], config.deallocate_pipeline_outputs)
        elif node.type == FuncType.RECV_BACKWARD:
            output_tensor_grads[microbatch] = p2p_communicator.recv_backward(
                send_tensor_shapes, p2p_communicator.is_pp_last_stage
            )
        elif node.type == FuncType.B:
            if microbatch == last_backward_microbatch:
                if config.grad_sync_func is None or p2p_communicator.is_pp_first_stage:
                    enable_grad_sync()
            with WeightGradStore.set_split_bw(True):
                input_tensor_grads[microbatch] = backward_step(
                    input_tensors.pop(microbatch),
                    output_tensors.pop(microbatch),
                    output_tensor_grads.pop(microbatch),
                    config,
                )
                WeightGradStore.flush()
        elif node.type == FuncType.W:
            WeightGradStore.pop()
        elif node.type == FuncType.SEND_BACKWARD:
            p2p_communicator.send_backward(input_tensor_grads.pop(microbatch), p2p_communicator.is_pp_first_stage)

    WeightGradStore.assert_empty()
    if no_sync_context is not None:
        enable_grad_sync()
        if config.grad_sync_func is not None:
            config.grad_sync_func(model.parameters())

    if config.finalize_model_grads_func is not None:
        finish_embedding_wgrad_compute(config, embedding_module, p2p_communicator.is_pp_last_stage, tp_group)
        config.finalize_model_grads_func(
            [model],
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if (
        hasattr(config, 'cuda_graph_impl')
        and config.cuda_graph_impl == 'local'
        and CudaGraphScope.full_iteration not in config.cuda_graph_scope
    ):
        create_cudagraphs()

    return forward_data_store
