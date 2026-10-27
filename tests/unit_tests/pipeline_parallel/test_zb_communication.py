"""Check local communication counts independently for each direction.

The validator does not verify microbatch identities, cross-rank peers, or runtime
ordering. One-sided inputs exercise count policy, not distributed boundaries.
"""

import pytest
from megatron.core.pipeline_parallel.zerobubble.scheduler.communication import validate_communication
from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import FuncType, ScheduledNode

pytestmark = [pytest.mark.sanity, pytest.mark.regression]


def _communication_nodes(direction, send_count, recv_count):
    return [
        ScheduledNode(FuncType[f"{operation}_{direction}"], microbatch, stage=1)
        for operation, count in (("SEND", send_count), ("RECV", recv_count))
        for microbatch in range(count)
    ]


@pytest.mark.parametrize(
    "nodes",
    [
        pytest.param([], id="empty"),
        pytest.param([("F", 0), ("B", 0), ("W", 0)], id="compute-only"),
        pytest.param(
            [
                ("RECV_FORWARD", 0),
                ("F", 0),
                ("SEND_FORWARD", 0),
                ("RECV_BACKWARD", 0),
                ("B", 0),
                ("SEND_BACKWARD", 0),
                ("W", 0),
            ],
            id="both-directions-equal-counts",
        ),
        pytest.param(
            [(kind, microbatch) for microbatch in range(3) for kind in ("RECV_FORWARD", "F", "SEND_FORWARD")],
            id="multiple-forward-equal-counts",
        ),
        pytest.param([("SEND_FORWARD", 0), ("SEND_FORWARD", 1)], id="send-forward-only"),
        pytest.param([("RECV_FORWARD", 0), ("RECV_FORWARD", 1)], id="recv-forward-only"),
        pytest.param([("SEND_BACKWARD", 0), ("SEND_BACKWARD", 1)], id="send-backward-only"),
        pytest.param([("RECV_BACKWARD", 0), ("RECV_BACKWARD", 1)], id="recv-backward-only"),
    ],
)
def test_accepts_local_count_patterns(nodes):
    schedule = [ScheduledNode(FuncType[kind], microbatch, stage=1) for kind, microbatch in nodes]
    before = tuple(schedule)

    assert validate_communication(schedule) is None
    assert tuple(schedule) == before


@pytest.mark.parametrize("direction", ["FORWARD", "BACKWARD"])
@pytest.mark.parametrize("send_count,recv_count", [(2, 1), (1, 3)])
def test_rejects_unequal_nonzero_counts(direction, send_count, recv_count):
    schedule = _communication_nodes(direction, send_count, recv_count)

    with pytest.raises(RuntimeError, match=f"SEND_{direction}={send_count}, RECV_{direction}={recv_count}"):
        validate_communication(schedule)


@pytest.mark.parametrize("direction,other_direction", [("FORWARD", "BACKWARD"), ("BACKWARD", "FORWARD")])
def test_rejects_imbalance_with_other_direction_balanced(direction, other_direction):
    schedule = _communication_nodes(direction, 2, 1) + _communication_nodes(other_direction, 3, 3)

    with pytest.raises(RuntimeError, match=f"SEND_{direction}=2, RECV_{direction}=1"):
        validate_communication(schedule)


def test_opposite_imbalances_do_not_cancel():
    schedule = _communication_nodes("FORWARD", 2, 1) + _communication_nodes("BACKWARD", 1, 2)

    with pytest.raises(RuntimeError, match="Unbalanced local ZB communication nodes"):
        validate_communication(schedule)
