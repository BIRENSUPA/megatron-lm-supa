# tests/unit_tests/pipeline_parallel/test_schedules_overlap.py
from unittest import mock

import pytest
import torch

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core import ModelParallelConfig
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import get_forward_backward_func
from tests.unit_tests.test_utilities import Utils

rank = Utils.rank

@pytest.mark.sanity
@pytest.mark.regression
@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pp_overlap_p2p_comm_interleaved():
    """Feature: pipeline-parallel comm-compute overlap (overlap_p2p_comm).

    Two ranks, PP2 / VP2 / TP1. The overlap path only exists in the interleaved
    schedule; with overlap_p2p_comm_warmup_flush=True the warmup issues async
    send/recv (overlap_p2p_comm=True) and stashes wait_handles. Completing the
    schedule without error and producing per-rank losses exercises the overlap
    comm path end-to-end.
    """
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
    )

    def forward_step_func(data_iterator, model):
        def loss_func(output_tensor):
            return rank, {"loss_reduced": rank}

        return torch.rand(512, 8, 256).cuda(), loss_func

    model = torch.nn.Linear(4, 1)

    def set_input_tensor(input_tensor):
        return None

    model.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_with_interleaving
    ), "overlap requires the interleaved schedule"

    config = ModelParallelConfig(
        pipeline_model_parallel_size=2,
        sequence_parallel=False,
        pipeline_dtype=torch.float,
        virtual_pipeline_model_parallel_size=2,
        overlap_p2p_comm=True,
        batch_p2p_comm=False,
        overlap_p2p_comm_warmup_flush=True,
    )
    config.hidden_size = 256
    model.config = config
    model.model_type = ModelType.encoder_or_decoder

    with mock.patch(
        "megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2
    ):
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=[range(0, 100), range(0, 100)],
            model=[model, model],
            num_microbatches=4,
            seq_length=512,
            micro_batch_size=8,
            decoder_seq_length=256,
            forward_only=True,
        )

    # Only the last PP stage reduces losses; earlier stages return an empty list.
    # The overlap comm path is exercised on every rank by completing the schedule
    # without error; we only validate loss values on the last stage.
    loss_reduced_expected = [{"loss_reduced": rank}, {"loss_reduced": rank}]
    for got, expected in zip(losses_reduced, loss_reduced_expected):
        assert got["loss_reduced"] == expected["loss_reduced"], "loss value does not match rank"

    Utils.destroy_model_parallel()

@pytest.mark.sanity
@pytest.mark.regression
@pytest.mark.internal
def test_pp_overlap_rejected_for_non_interleaved():
    """Feature: pipeline-parallel comm-compute overlap guard.

    overlap_p2p_comm is only supported by the interleaved schedule; the
    non-interleaved schedule must reject it. Runs single-rank (no comm).
    """
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
    )

    config = ModelParallelConfig(
        pipeline_model_parallel_size=2,
        sequence_parallel=False,
        pipeline_dtype=torch.float,
        overlap_p2p_comm=True,
        batch_p2p_comm=False,
    )
    config.hidden_size = 16

    model = torch.nn.Linear(4, 1)
    model.config = config
    model.model_type = ModelType.encoder_or_decoder
    model.set_input_tensor = lambda x: None

    def forward_step_func(data_iterator, model):
        return torch.rand(4, 1).cuda(), lambda o: (0, {"loss_reduced": 0})

    with pytest.raises(ValueError):
        schedule.forward_backward_pipelining_without_interleaving(
            forward_step_func=forward_step_func,
            data_iterator=None,
            model=[model],
            num_microbatches=1,
            seq_length=4,
            micro_batch_size=1,
            forward_only=True,
        )

    Utils.destroy_model_parallel()
