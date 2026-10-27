"""Unit tests for zero-bubble scheduling, dependencies, memory and caching."""

import copy
from itertools import product
from unittest.mock import Mock

import pytest
import torch
from megatron.core.pipeline_parallel.zerobubble.scheduler import zb
from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import FuncType, GraphConfig, ScheduledNode

pytestmark = [pytest.mark.sanity, pytest.mark.regression]


@pytest.fixture
def make_graph_config():
    def make(n_stages=2, n_micro=4, **overrides):
        values = dict(
            n_stages=n_stages,
            n_micro=n_micro,
            cost_f=[1.0] * n_stages,
            cost_b=[1.0] * n_stages,
            cost_w=[1.0] * n_stages,
            cost_comm=0.0,
            mem_f=[2.0] * n_stages,
            mem_b=[-1.0] * n_stages,
            mem_w=[-1.0] * n_stages,
        )
        values.update(overrides)
        return GraphConfig(**values)

    return make


@pytest.fixture(autouse=True)
def isolate_schedule_cache(monkeypatch):
    monkeypatch.setattr(zb, "_SCHEDULE_CACHE", {})


def _assert_valid_manual_order(graph, order, completion, duration):
    """Check the schedule and return its per-stage memory traces."""
    config = graph.config
    assert len(order) == len(completion) == 3 * graph.nstages * graph.nmb
    spans, memory_traces = [], []
    for stage in range(graph.nstages):
        nodes = [node for node in range(graph.nnodes) if graph.get_stage(node) == stage]
        assert sorted(order[node] for node in nodes) == list(range(3 * graph.nmb))
        nodes.sort(key=lambda node: order[node])
        memory, last_end = 0, 0
        memory_trace = []
        first_start = completion[nodes[0]] - graph.get_cost(nodes[0])
        limit = config.max_mem[stage] if config.max_mem is not None else config.mem_f[stage] * graph.nmb * 3
        for node in nodes:
            start = completion[node] - graph.get_cost(node)
            assert start >= last_end
            for parent in graph.parents[node]:
                delay = config.cost_comm if graph.get_stage(parent) != stage else 0
                assert start >= completion[parent] + delay
            memory += graph.get_mem(node)
            assert 0 <= memory <= limit
            memory_trace.append(memory)
            last_end = completion[node]
        assert memory == 0
        # Check the local F -> B -> W dependency independently of graph.parents.
        for microbatch in range(graph.nmb):
            forward = graph.get_id(0, stage, microbatch)
            backward = graph.get_id(1, stage, microbatch)
            weight = graph.get_id(2, stage, microbatch)
            assert completion[backward] - config.cost_b[stage] >= completion[forward]
            assert completion[weight] - config.cost_w[stage] >= completion[backward]
        spans.append(last_end - first_start)
        memory_traces.append(memory_trace)
    assert duration == max(spans)
    return memory_traces


def test_graph_ids_dependencies_and_stage_costs(make_graph_config):
    config = make_graph_config(
        n_micro=2,
        cost_f=[1.0, 2.0],
        cost_b=[3.0, 4.0],
        cost_w=[5.0, 6.0],
        mem_f=[2.0, 3.0],
        mem_b=[-1.0, -2.0],
        mem_w=[-1.0, -1.0],
    )
    graph = zb.Graph.build_graph(2, 2, config)
    assert graph.nnodes == 12
    assert graph.parents == [set(), {0}, {0}, {1, 2}, {6}, {4, 7}, {2}, {3, 6}, {4}, {5, 8}, {6}, {7, 10}]
    assert graph.name == ["F0", "F1", "F0", "F1", "B0", "B1", "B0", "B1", "W0", "W1", "W0", "W1"]
    for kind, (costs, memory) in enumerate(
        zip((config.cost_f, config.cost_b, config.cost_w), (config.mem_f, config.mem_b, config.mem_w))
    ):
        for stage in range(2):
            for microbatch in range(2):
                node = graph.get_id(kind, stage, microbatch)
                assert node == kind * 4 + stage * 2 + microbatch
                assert graph.get_stage(node) == stage
                assert graph.get_cost(node) == costs[stage]
                assert graph.get_mem(node) == memory[stage]


def test_requires_order_only_for_unordered_nodes_on_same_stage(make_graph_config):
    graph = zb.Graph.build_graph(2, 2, make_graph_config(n_micro=2))
    graph.precede = torch.zeros((graph.nnodes, graph.nnodes), dtype=torch.bool, device="cpu")
    assert not graph.requires_order(0, 0)
    assert not graph.requires_order(0, 2)
    assert graph.requires_order(0, 1)
    graph.precede[0][1] = True
    assert not graph.requires_order(0, 1)
    assert not graph.requires_order(1, 0)


@pytest.mark.parametrize("flags", list(product([False, True], repeat=3)))
@pytest.mark.parametrize(
    "stages,microbatches,overrides",
    [
        (1, 1, {}),
        (2, 4, {}),
        (3, 8, {"max_mem": [6.0] * 3}),
        (4, 12, {"max_mem": [8.0] * 4, "cost_comm": 0.5}),
        (3, 6, {"cost_f": [1.0, 2.0, 1.5], "cost_b": [2.0, 1.0, 3.0], "cost_comm": 0.25}),
    ],
    ids=["single", "default-memory", "bounded-memory", "communication-delay", "unequal-stage-costs"],
)
def test_manual_order_respects_dependencies_timing_and_memory(
    make_graph_config, stages, microbatches, overrides, flags
):
    config = make_graph_config(stages, microbatches, **overrides)
    graph = zb.Graph.build_graph(stages, microbatches, config)
    _assert_valid_manual_order(graph, *graph.manual_order(*flags))


def test_two_stage_one_microbatch_has_known_timeline(make_graph_config):
    graph = zb.Graph.build_graph(2, 1, make_graph_config(n_micro=1))
    order, completion, duration = graph.manual_order()
    assert order == [0, 0, 1, 1, 2, 2]
    assert completion == [1.0, 2.0, 4.0, 3.0, 5.0, 4.0]
    assert duration == 5.0


@pytest.mark.parametrize("limit,reclaim_before_forward", [(2.0, True), (3.0, False)])
def test_forward_memory_limit_controls_weight_reclamation(make_graph_config, limit, reclaim_before_forward):
    config = make_graph_config(1, 2, max_mem=[limit])
    graph = zb.Graph.build_graph(1, 2, config)
    order, completion, duration = graph.manual_order()
    memory_traces = _assert_valid_manual_order(graph, order, completion, duration)
    forward = graph.get_id(0, 0, 1)
    weight = graph.get_id(2, 0, 0)
    assert (order[weight] < order[forward]) == reclaim_before_forward
    assert max(memory_traces[0]) == limit


def test_backward_reclaims_weights_when_it_needs_memory(make_graph_config):
    config = make_graph_config(1, 2, mem_f=[1.0], mem_b=[1.0], mem_w=[-2.0], max_mem=[3.0])
    graph = zb.Graph.build_graph(1, 2, config)
    order, completion, duration = graph.manual_order()
    memory_traces = _assert_valid_manual_order(graph, order, completion, duration)
    nodes = sorted(range(graph.nnodes), key=lambda node: order[node])
    assert [graph.name[node] for node in nodes] == ["F0", "B0", "F1", "W0", "B1", "W1"]
    assert memory_traces == [[1.0, 2.0, 3.0, 1.0, 2.0, 0.0]]


@pytest.mark.parametrize("allow_bubble", [True, False])
def test_allow_bubble_controls_forward_warmup_before_first_backward(make_graph_config, allow_bubble):
    config = make_graph_config(2, 2, cost_f=[2.0, 1.0], cost_b=[1.0, 0.5])
    graph = zb.Graph.build_graph(2, 2, config)
    order, completion, duration = graph.manual_order(allow_bubble_before_first_b=allow_bubble, prioritize_b=False)
    _assert_valid_manual_order(graph, order, completion, duration)
    forward = graph.get_id(0, 0, 1)
    backward = graph.get_id(1, 0, 0)
    assert (order[backward] < order[forward]) == allow_bubble


def test_reports_memory_that_cannot_fit_a_later_forward(make_graph_config):
    # Initial placement is unchecked; the error comes from a later forward.
    config = make_graph_config(max_mem=[1.0, 1.0])
    graph = zb.Graph.build_graph(2, 4, config)
    with pytest.raises(ValueError, match="Cannot fit memory"):
        graph.manual_order()


def test_initial_solution_selects_best_candidate_and_keeps_first_tie(capsys):
    costs = [9, 7, 3, 5, 8, 3, 6, 4]
    graph = Mock()
    graph.manual_order.side_effect = [([index], [index + 10], cost) for index, cost in enumerate(costs)]
    assert zb.initial_solution(graph, print_result=False) == (3, [2], [12])
    calls = graph.manual_order.call_args_list
    assert len(calls) == 8
    flag_names = ("allow_bubble_before_first_b", "prioritize_b", "no_bubble_greedy")
    assert {tuple(call.kwargs[name] for name in flag_names) for call in calls} == set(product([True, False], repeat=3))
    assert capsys.readouterr().out == ""


def test_printed_schedule_labels_two_digit_microbatches(make_graph_config, capsys):
    config = make_graph_config(1, 11, cost_f=[5.0], cost_b=[5.0], cost_w=[5.0])
    graph = zb.Graph.build_graph(1, 11, config)
    duration, _, _ = zb.initial_solution(graph)
    output = capsys.readouterr().out
    for label in ("F0--F", "B0--B", "W0--W", "F10-F", "B10-B", "W10-W"):
        assert label in output
    assert "Longest stage time:  165.0" in output
    assert duration == 165.0


def test_scheduled_nodes_are_sorted_by_completion_and_keep_metadata(make_graph_config):
    graph = zb.Graph.build_graph(2, 2, make_graph_config(n_micro=2))
    schedules = zb.create_scheduled_nodes(graph, [1, 3, 2, 5, 2, 5, 4, 6, 4, 6, 5.5, 7])
    assert [[(node.type, node.microbatch) for node in stage] for stage in schedules] == [
        [(FuncType.F, 0), (FuncType.B, 0), (FuncType.F, 1), (FuncType.W, 0), (FuncType.B, 1), (FuncType.W, 1)],
        [(FuncType.F, 0), (FuncType.B, 0), (FuncType.F, 1), (FuncType.W, 0), (FuncType.B, 1), (FuncType.W, 1)],
    ]
    for stage, nodes in enumerate(schedules):
        assert all(node.stage == node.layer_group_idx == stage for node in nodes)


@pytest.mark.parametrize("stages,microbatches", [(1, 1), (2, 4), (4, 8)])
def test_create_schedule_contains_every_computation_in_dependency_order(make_graph_config, stages, microbatches):
    schedules = zb.create_schedule(make_graph_config(stages, microbatches), print_result=False)
    assert len(schedules) == stages
    for stage, schedule in enumerate(schedules):
        assert len(schedule) == 3 * microbatches
        positions = {(node.type, node.microbatch): index for index, node in enumerate(schedule)}
        assert set(positions) == set(product([FuncType.F, FuncType.B, FuncType.W], range(microbatches)))
        assert all(node.stage == node.layer_group_idx == stage for node in schedule)
        for microbatch in range(microbatches):
            assert positions[FuncType.F, microbatch] < positions[FuncType.B, microbatch]
            assert positions[FuncType.B, microbatch] < positions[FuncType.W, microbatch]
        for kind in (FuncType.F, FuncType.B, FuncType.W):
            assert [node.microbatch for node in schedule if node.type == kind] == list(range(microbatches))


def test_point_to_point_nodes_surround_computation_without_changing_it():
    compute = [
        ScheduledNode(kind, microbatch=3, stage=1, layer_group_idx=1) for kind in (FuncType.F, FuncType.B, FuncType.W)
    ]
    schedule = zb.add_point_to_point_nodes(compute)
    assert [node.type for node in schedule] == [
        FuncType.RECV_FORWARD,
        FuncType.F,
        FuncType.SEND_FORWARD,
        FuncType.RECV_BACKWARD,
        FuncType.B,
        FuncType.SEND_BACKWARD,
        FuncType.W,
    ]
    assert schedule[1] is compute[0] and schedule[4] is compute[1] and schedule[6] is compute[2]
    assert all(node.microbatch == 3 and node.stage == 1 for node in schedule)
    assert all(node.layer_group_idx == 1 for node in compute)
    assert all(schedule[index].layer_group_idx is None for index in (0, 2, 3, 5))
    assert zb.add_point_to_point_nodes([]) == []


def test_schedule_cache_shares_generation_across_ranks_and_separates_dimensions(monkeypatch):
    create = Mock(wraps=zb.create_schedule)
    monkeypatch.setattr(zb, "create_schedule", create)
    rank0 = zb.get_zb_schedule(4, 2, 0)
    rank1 = zb.get_zb_schedule(4, 2, 1)
    assert zb.get_zb_schedule(4, 2, 0) is rank0
    assert create.call_count == 1
    assert len(rank0) == len(rank1) == 28
    assert all(node.stage == 0 for node in rank0)
    assert all(node.stage == 1 for node in rank1)
    assert len(zb.get_zb_schedule(2, 2, 0)) == 14
    assert all(node.stage == 2 for node in zb.get_zb_schedule(4, 3, 2))
    assert create.call_count == 3
    assert len(zb._SCHEDULE_CACHE) == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("cost_f", [2.0, 1.0]),
        ("cost_b", [2.0, 1.0]),
        ("cost_w", [2.0, 1.0]),
        ("cost_comm", 0.5),
        ("mem_f", [3.0, 2.0]),
        ("mem_b", [-2.0, -1.0]),
        ("mem_w", [-2.0, -1.0]),
        ("max_mem", [4.0, 4.0]),
        ("print_scaling", 2),
        ("max_chunks", 2),
        ("n_stages", 3),
        ("n_micro", 5),
    ],
)
def test_cache_key_includes_configuration_values(make_graph_config, field, value):
    config = make_graph_config()
    equivalent = copy.deepcopy(config)
    original_key = zb._get_cache_key(config)
    assert original_key == zb._get_cache_key(equivalent)
    setattr(equivalent, field, value)
    assert zb._get_cache_key(equivalent) != original_key
    assert zb._get_cache_key(config) == original_key


def test_cache_key_snapshots_configuration_lists(make_graph_config):
    config = make_graph_config()
    original_key = zb._get_cache_key(config)
    original_hash = hash(original_key)
    original_values = copy.deepcopy(original_key)
    config.cost_f[0] = 2.0
    assert zb._get_cache_key(config) != original_key
    assert original_key == original_values
    assert hash(original_key) == original_hash
