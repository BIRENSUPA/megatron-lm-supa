"""ZB schedule passes validate communication and preserve the input schedule."""

from unittest import mock

import pytest
from megatron.core.pipeline_parallel.zerobubble.scheduler import passes
from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import FuncType, ScheduledNode

pytestmark = [pytest.mark.sanity, pytest.mark.regression]


@pytest.mark.parametrize("empty", [True, False], ids=["empty", "full-schedule"])
def test_validates_once_and_preserves_schedule_and_nodes(empty):
    schedule = (
        []
        if empty
        else [
            ScheduledNode(kind, microbatch=0, stage=1, layer_group_idx=1)
            for kind in (
                FuncType.RECV_FORWARD,
                FuncType.F,
                FuncType.SEND_FORWARD,
                FuncType.RECV_BACKWARD,
                FuncType.B,
                FuncType.SEND_BACKWARD,
                FuncType.W,
            )
        ]
    )
    before = tuple(schedule)

    with mock.patch.object(passes, "validate_communication", wraps=passes.validate_communication) as validate:
        result = passes.run_schedule_passes(schedule)

    validate.assert_called_once_with(schedule)
    assert validate.call_args.args[0] is schedule
    assert result is schedule
    assert len(schedule) == len(before)
    assert all(node is original for node, original in zip(schedule, before))


def test_propagates_real_communication_error_without_changing_schedule():
    schedule = [
        ScheduledNode(FuncType.SEND_BACKWARD, microbatch=0, stage=1),
        ScheduledNode(FuncType.SEND_BACKWARD, microbatch=1, stage=1),
        ScheduledNode(FuncType.RECV_BACKWARD, microbatch=0, stage=1),
    ]
    before = tuple(schedule)

    with pytest.raises(RuntimeError, match="SEND_BACKWARD=2, RECV_BACKWARD=1"):
        passes.run_schedule_passes(schedule)

    assert len(schedule) == len(before)
    assert all(node is original for node, original in zip(schedule, before))


def test_propagates_original_validation_exception():
    schedule = [ScheduledNode(FuncType.F, microbatch=0, stage=1)]
    before = tuple(schedule)
    error = RuntimeError("validator sentinel")

    with mock.patch.object(passes, "validate_communication", side_effect=error) as validate:
        with pytest.raises(RuntimeError) as exc_info:
            passes.run_schedule_passes(schedule)

    assert exc_info.value is error
    validate.assert_called_once_with(schedule)
    assert validate.call_args.args[0] is schedule
    assert len(schedule) == len(before)
    assert all(node is original for node, original in zip(schedule, before))
