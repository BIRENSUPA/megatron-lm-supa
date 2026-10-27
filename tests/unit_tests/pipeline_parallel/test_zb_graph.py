"""Contracts for zero-bubble graph configuration and scheduled nodes."""

from dataclasses import FrozenInstanceError

import pytest
from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import FuncType, GraphConfig, ScheduledNode

pytestmark = [pytest.mark.sanity, pytest.mark.regression]


@pytest.fixture
def config_values():
    return dict(
        mem_f=[2.0, 3.0],
        mem_b=[-1.0, -2.0],
        mem_w=[-1.0, -1.0],
        cost_f=[1.0, 2.0],
        cost_b=[3.0, 4.0],
        cost_w=[5.0, 6.0],
        n_stages=2,
        n_micro=4,
    )


def test_graph_config_retains_fields_and_optional_defaults(config_values):
    config = GraphConfig(**config_values)
    for field, value in config_values.items():
        assert getattr(config, field) == value
    assert config.max_mem is None
    assert config.cost_comm == 0.0
    assert config.print_scaling == 1
    assert config.max_chunks == 1


def test_graph_config_retains_explicit_optional_fields(config_values):
    config_values.update(max_mem=[8.0, 12.0], cost_comm=0.25, print_scaling=2, max_chunks=2)
    config = GraphConfig(**config_values)
    for field, value in config_values.items():
        assert getattr(config, field) == value


@pytest.mark.parametrize("field", ["cost_f", "cost_b", "cost_w", "cost_comm"])
@pytest.mark.parametrize("invalid", [1, "1.0"], ids=["integer", "string"])
def test_graph_config_requires_float_costs(config_values, field, invalid):
    config_values[field] = invalid if field == "cost_comm" else [1.0, invalid]
    with pytest.raises(AssertionError):
        GraphConfig(**config_values)


def test_graph_config_rejects_unbalanced_stage_memory(config_values):
    config_values["mem_w"][1] = 0.0
    with pytest.raises(AssertionError):
        GraphConfig(**config_values)


@pytest.mark.parametrize("field", ["n_stages", "n_micro"])
def test_graph_config_requires_stage_and_microbatch_counts(config_values, field):
    config_values[field] = None
    with pytest.raises(AssertionError):
        GraphConfig(**config_values)


def test_func_type_has_seven_distinct_node_kinds():
    expected_names = {"F", "B", "W", "SEND_FORWARD", "RECV_FORWARD", "SEND_BACKWARD", "RECV_BACKWARD"}
    assert set(FuncType.__members__) == expected_names
    assert len(list(FuncType)) == 7
    assert len({kind.value for kind in FuncType}) == 7


def test_scheduled_node_retains_required_and_optional_fields():
    minimal = ScheduledNode(FuncType.F, microbatch=3)
    assert minimal.type is FuncType.F
    assert minimal.microbatch == 3
    assert minimal.stage is None
    assert minimal.layer_group_idx is None
    assigned = ScheduledNode(FuncType.B, microbatch=4, stage=1, layer_group_idx=2)
    assert assigned.type is FuncType.B
    assert assigned.microbatch == 4
    assert assigned.stage == 1
    assert assigned.layer_group_idx == 2


@pytest.mark.parametrize("values", [{"microbatch": 0}, {"type": FuncType.F}])
def test_scheduled_node_requires_type_and_microbatch(values):
    with pytest.raises(TypeError):
        ScheduledNode(**values)


@pytest.mark.parametrize("field,value", [("type", FuncType.B), ("microbatch", 1), ("stage", 1), ("layer_group_idx", 1)])
def test_scheduled_node_equality_compares_values(field, value):
    values = dict(type=FuncType.F, microbatch=0, stage=0, layer_group_idx=0)
    node = ScheduledNode(**values)
    assert node == ScheduledNode(**values)
    values[field] = value
    assert node != ScheduledNode(**values)


def test_scheduled_node_is_immutable():
    node = ScheduledNode(FuncType.F, microbatch=0)
    with pytest.raises(FrozenInstanceError):
        node.microbatch = 1
    assert node.microbatch == 0
