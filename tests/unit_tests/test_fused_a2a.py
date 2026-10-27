"""Test the SUPA buffer policy without allocating a DeepEP communication buffer."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from megatron.core.transformer.moe import fused_a2a as core_fused_a2a

from megatron_supa.core.transformer.moe import fused_a2a

pytestmark = [pytest.mark.sanity, pytest.mark.regression]


@pytest.mark.parametrize("group_size,device_count,expected", [(2, 8, 2), (8, 2, 2), (8, 8, 8)])
def test_ranks_per_node_is_limited_by_group_and_visible_devices(monkeypatch, group_size, device_count, expected):
    monkeypatch.delenv("MEGATRON_NUM_RANKS_PER_NODE", raising=False)
    monkeypatch.setattr(fused_a2a.torch.cuda, "device_count", lambda: device_count)

    assert fused_a2a.get_num_ranks_per_node(group_size) == expected


def test_empty_ranks_override_uses_visible_devices(monkeypatch):
    monkeypatch.setenv("MEGATRON_NUM_RANKS_PER_NODE", "")
    monkeypatch.setattr(fused_a2a.torch.cuda, "device_count", lambda: 3)

    assert fused_a2a.get_num_ranks_per_node(8) == 3


@pytest.mark.parametrize("group_size,override,expected", [(8, "2", 2), (2, "8", 2), (8, "1", 1)])
def test_ranks_override_is_limited_by_group_without_querying_devices(monkeypatch, group_size, override, expected):
    monkeypatch.setenv("MEGATRON_NUM_RANKS_PER_NODE", override)
    device_count = Mock(side_effect=AssertionError("an explicit override must not query local devices"))
    monkeypatch.setattr(fused_a2a.torch.cuda, "device_count", device_count)

    assert fused_a2a.get_num_ranks_per_node(group_size) == expected


@pytest.mark.parametrize("override", ["0", "-1"])
def test_nonpositive_ranks_override_is_rejected(monkeypatch, override):
    monkeypatch.setenv("MEGATRON_NUM_RANKS_PER_NODE", override)

    with pytest.raises(ValueError, match="must be positive"):
        fused_a2a.get_num_ranks_per_node(8)


def test_noninteger_ranks_override_is_rejected(monkeypatch):
    monkeypatch.setenv("MEGATRON_NUM_RANKS_PER_NODE", "not-an-integer")

    with pytest.raises(ValueError):
        fused_a2a.get_num_ranks_per_node(8)


@pytest.fixture
def buffer_policy(monkeypatch):
    # DeepEP owns device allocation/communication; exercise only our size/cache policy.
    dispatch = SimpleNamespace(
        get_nvl_buffer_size_hint=Mock(return_value=128),
        get_rdma_buffer_size_hint=Mock(return_value=256),
    )
    combine = SimpleNamespace(
        get_nvl_buffer_size_hint=Mock(return_value=64),
        get_rdma_buffer_size_hint=Mock(return_value=512),
    )
    buffer_type = Mock(
        side_effect=lambda group, nvl, rdma: SimpleNamespace(group=group, num_nvl_bytes=nvl, num_rdma_bytes=rdma)
    )
    buffer_type.get_dispatch_config.return_value = dispatch
    buffer_type.get_combine_config.return_value = combine
    deep_ep = ModuleType("deep_ep")
    deep_ep.Buffer = buffer_type
    monkeypatch.setitem(sys.modules, "deep_ep", deep_ep)
    monkeypatch.setattr(core_fused_a2a, "_buffer", None)
    monkeypatch.setenv("MEGATRON_NUM_RANKS_PER_NODE", "2")
    group = Mock(spec=["size"])
    group.size.return_value = 8
    return SimpleNamespace(group=group, buffer_type=buffer_type, dispatch=dispatch, combine=combine)


def test_buffer_reserves_maximum_dispatch_and_combine_capacity(buffer_policy):
    buffer = fused_a2a.get_buffer(buffer_policy.group, hidden_bytes=4096)

    assert buffer is core_fused_a2a._buffer
    assert buffer.group is buffer_policy.group
    assert buffer.num_nvl_bytes == 128
    assert buffer.num_rdma_bytes == 512
    buffer_policy.buffer_type.assert_called_once_with(buffer_policy.group, 128, 512)


def test_buffer_size_hints_receive_group_size_and_ranks_per_node(buffer_policy):
    fused_a2a.get_buffer(buffer_policy.group, hidden_bytes=4096)

    buffer_policy.buffer_type.get_dispatch_config.assert_called_once_with(8)
    buffer_policy.buffer_type.get_combine_config.assert_called_once_with(8)
    for config in (buffer_policy.dispatch, buffer_policy.combine):
        config.get_nvl_buffer_size_hint.assert_called_once_with(4096, 8, 2)
        config.get_rdma_buffer_size_hint.assert_called_once_with(4096, 8, 2)


@pytest.mark.parametrize("nvl,rdma", [(128, 512), (256, 1024)])
def test_sufficient_same_group_buffer_is_reused(buffer_policy, monkeypatch, nvl, rdma):
    cached = SimpleNamespace(group=buffer_policy.group, num_nvl_bytes=nvl, num_rdma_bytes=rdma)
    monkeypatch.setattr(core_fused_a2a, "_buffer", cached)

    assert fused_a2a.get_buffer(buffer_policy.group, hidden_bytes=4096) is cached
    assert core_fused_a2a._buffer is cached
    buffer_policy.buffer_type.assert_not_called()


@pytest.mark.parametrize("reason", ["different-group", "insufficient-nvl", "insufficient-rdma"])
def test_incompatible_buffer_is_replaced(buffer_policy, monkeypatch, reason):
    cached = SimpleNamespace(
        group=object() if reason == "different-group" else buffer_policy.group,
        num_nvl_bytes=127 if reason == "insufficient-nvl" else 128,
        num_rdma_bytes=511 if reason == "insufficient-rdma" else 512,
    )
    monkeypatch.setattr(core_fused_a2a, "_buffer", cached)

    buffer = fused_a2a.get_buffer(buffer_policy.group, hidden_bytes=4096)

    assert buffer is not cached
    assert core_fused_a2a._buffer is buffer
    assert buffer.group is buffer_policy.group
    assert buffer.num_nvl_bytes == 128
    assert buffer.num_rdma_bytes == 512


def test_buffer_is_grown_after_a_larger_request(buffer_policy):
    first = fused_a2a.get_buffer(buffer_policy.group, hidden_bytes=4096)
    buffer_policy.dispatch.get_nvl_buffer_size_hint.return_value = 256

    second = fused_a2a.get_buffer(buffer_policy.group, hidden_bytes=8192)

    assert second is not first
    assert first.num_nvl_bytes == 128
    assert second.num_nvl_bytes == 256
    assert second.num_rdma_bytes == 512
    assert core_fused_a2a._buffer is second
