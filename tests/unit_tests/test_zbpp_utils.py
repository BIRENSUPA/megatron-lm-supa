"""State and callback tests for deferred weight-gradient computation."""

import queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from megatron.core import zbpp_utils

pytestmark = [pytest.mark.sanity, pytest.mark.regression]


@pytest.fixture
def supported_args():
    return SimpleNamespace(
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=None,
        overlap_grad_reduce=False,
        gradient_accumulation_fusion=True,
        transformer_impl="local",
    )


@pytest.fixture
def store(monkeypatch, supported_args):
    store = zbpp_utils.WeightGradStore
    monkeypatch.setattr(store, "should_split_bw", False)
    monkeypatch.setattr(store, "cache", [])
    monkeypatch.setattr(store, "weight_grad_queue", None)
    monkeypatch.setattr(zbpp_utils, "get_args", lambda: supported_args)
    monkeypatch.setattr(zbpp_utils.parallel_state, "get_virtual_pipeline_model_parallel_world_size", lambda: None)
    monkeypatch.setattr(zbpp_utils.parallel_state, "get_pipeline_model_parallel_rank", lambda: 7)
    return store


def _gradient_callbacks(weight, amount, label, events):
    def pre_func(*, async_op):
        assert async_op is False
        events.append(("pre", label, async_op))
        return weight, amount

    def func(target, gradient):
        assert target is weight
        assert gradient == amount
        events.append(("func", label, gradient))
        target["grad"] += gradient

    return pre_func, func


def test_is_supported_with_standard_pipeline_configuration(store):
    assert store.is_supported() is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("pipeline_model_parallel_size", 1),
        ("virtual_pipeline_model_parallel_size", 2),
        ("overlap_grad_reduce", True),
        ("gradient_accumulation_fusion", False),
        ("transformer_impl", "transformer_engine"),
    ],
)
def test_is_supported_rejects_each_unsupported_configuration(store, supported_args, field, value):
    setattr(supported_args, field, value)
    assert store.is_supported() is False


def test_split_bw_disabled_does_not_read_arguments(store, monkeypatch):
    get_args = Mock(side_effect=AssertionError("disabled split must not read arguments"))
    monkeypatch.setattr(zbpp_utils, "get_args", get_args)
    assert store.split_bw() is False
    get_args.assert_not_called()


def test_split_bw_enabled_checks_actual_configuration(store, supported_args, monkeypatch):
    get_args = Mock(return_value=supported_args)
    monkeypatch.setattr(zbpp_utils, "get_args", get_args)
    with store.set_split_bw(True):
        assert store.split_bw() is True
        supported_args.gradient_accumulation_fusion = False
        assert store.split_bw() is False
    assert get_args.call_count == 2


def test_set_split_bw_restores_each_nested_context(store):
    assert store.should_split_bw is False
    with store.set_split_bw(True):
        assert store.should_split_bw is True
        with store.set_split_bw(False):
            assert store.should_split_bw is False
        assert store.should_split_bw is True
    assert store.should_split_bw is False


@pytest.mark.parametrize("initial", [False, True])
def test_set_split_bw_restores_state_after_exception(store, initial):
    store.should_split_bw = initial
    with pytest.raises(ValueError, match="context body failed"):
        with store.set_split_bw(not initial):
            assert store.should_split_bw is not initial
            raise ValueError("context body failed")
    assert store.should_split_bw is initial


@pytest.mark.parametrize("enabled,supported", [(False, True), (True, False)])
def test_put_rejects_disabled_or_unsupported_mode(store, supported_args, enabled, supported):
    supported_args.gradient_accumulation_fusion = supported
    weight, events = {"grad": 0}, []
    pre_func, func = _gradient_callbacks(weight, 3, "rejected", events)
    with store.set_split_bw(enabled):
        with pytest.raises(AssertionError):
            store.put(weight, pre_func, func)
    assert store.cache == []
    assert weight["grad"] == 0
    assert events == []


def test_put_caches_callbacks_without_executing_them(store):
    weight, events = {"grad": 0}, []
    pre_func, func = _gradient_callbacks(weight, 3, "pending", events)
    with store.set_split_bw(True):
        store.put(weight, pre_func, func)
    assert len(store.cache) == 1
    cached_weight, cached_pre_func, cached_func = store.cache[0]
    assert cached_weight is weight
    assert cached_pre_func is pre_func
    assert cached_func is func
    assert weight["grad"] == 0
    assert events == []
    assert store.weight_grad_queue is None


@pytest.mark.parametrize("num_chunks,expected_chunks", [(None, 1), (1, 1), (2, 2)])
def test_lazy_init_creates_chunk_queues_once(store, monkeypatch, num_chunks, expected_chunks):
    monkeypatch.setattr(zbpp_utils.parallel_state, "get_virtual_pipeline_model_parallel_world_size", lambda: num_chunks)
    store.lazy_init()
    queues = store.weight_grad_queue
    assert len(queues) == expected_chunks
    assert all(len(chunk_queues) == 1 for chunk_queues in queues)
    assert all(isinstance(chunk_queues[0], queue.Queue) and chunk_queues[0].empty() for chunk_queues in queues)
    assert len({id(chunk_queues[0]) for chunk_queues in queues}) == expected_chunks
    get_chunks = Mock(side_effect=AssertionError("initialized queues must not be recreated"))
    monkeypatch.setattr(zbpp_utils.parallel_state, "get_virtual_pipeline_model_parallel_world_size", get_chunks)
    store.lazy_init()
    assert store.weight_grad_queue is queues
    get_chunks.assert_not_called()


def test_flush_and_pop_preserve_batches_and_execute_callbacks_in_fifo_order(store):
    weight, events = {"grad": 0}, []
    with store.set_split_bw(True):
        for label, amount in (("first", 2), ("second", 3)):
            pre_func, func = _gradient_callbacks(weight, amount, label, events)
            store.put(weight, pre_func, func)
        first_batch = store.cache
        store.flush()
        assert store.cache == [] and store.cache is not first_batch
        assert len(first_batch) == 2

        pre_func, func = _gradient_callbacks(weight, 7, "third", events)
        store.put(weight, pre_func, func)
        second_batch = store.cache
        store.flush()
        assert store.cache == [] and store.cache is not second_batch
        assert len(second_batch) == 1
        assert len(first_batch) == 2
        assert events == []
        assert weight["grad"] == 0
        assert store.weight_grad_queue[0][0].qsize() == 2

        store.pop()
        assert weight["grad"] == 5
        assert store.weight_grad_queue[0][0].qsize() == 1
        assert events == [
            ("pre", "first", False),
            ("func", "first", 2),
            ("pre", "second", False),
            ("func", "second", 3),
        ]
        store.pop()
        assert weight["grad"] == 12
        assert events[-2:] == [("pre", "third", False), ("func", "third", 7)]
        assert len(events) == 6
        store.assert_empty()


def test_flush_disabled_with_empty_cache_does_not_enqueue(store):
    store.flush()
    assert store.cache == []
    assert store.weight_grad_queue[0][0].empty()


def test_flush_disabled_rejects_pending_cache(store):
    weight, events = {"grad": 0}, []
    pre_func, func = _gradient_callbacks(weight, 3, "pending", events)
    with store.set_split_bw(True):
        store.put(weight, pre_func, func)
    pending = store.cache
    with pytest.raises(AssertionError):
        store.flush()
    assert store.cache is pending and len(pending) == 1
    assert store.weight_grad_queue[0][0].empty()
    assert events == []


def test_pop_empty_queue_reports_pipeline_rank(store, monkeypatch):
    get_rank = Mock(return_value=7)
    monkeypatch.setattr(zbpp_utils.parallel_state, "get_pipeline_model_parallel_rank", get_rank)
    with pytest.raises(RuntimeError, match="Pop empty weight grad queue on pipeline rank 7"):
        store.pop()
    get_rank.assert_called_once_with()


def test_clear_drains_only_the_selected_chunk_and_is_repeatable(store, monkeypatch):
    """Check queue isolation; virtual-pipeline runtime support is a separate restriction."""
    monkeypatch.setattr(zbpp_utils.parallel_state, "get_virtual_pipeline_model_parallel_world_size", lambda: 2)
    weights, events = [{"grad": 0}, {"grad": 0}], []
    with store.set_split_bw(True):
        for chunk, label, amount in ((0, "first", 2), (1, "other", 11), (0, "second", 3)):
            pre_func, func = _gradient_callbacks(weights[chunk], amount, label, events)
            store.put(weights[chunk], pre_func, func)
            store.flush(chunk=chunk)
    store.clear(chunk=0)
    assert weights == [{"grad": 5}, {"grad": 0}]
    assert store.weight_grad_queue[0][0].empty()
    assert store.weight_grad_queue[1][0].qsize() == 1
    assert events == [
        ("pre", "first", False),
        ("func", "first", 2),
        ("pre", "second", False),
        ("func", "second", 3),
    ]
    store.clear(chunk=0)
    assert len(events) == 4
    assert weights == [{"grad": 5}, {"grad": 0}]
    store.clear(chunk=1)
    assert weights == [{"grad": 5}, {"grad": 11}]
    assert events[-2:] == [("pre", "other", False), ("func", "other", 11)]
    assert len(events) == 6
    store.assert_empty()


@pytest.mark.parametrize("initialized", [False, True])
def test_assert_empty_accepts_uninitialized_or_empty_queues(store, initialized):
    if initialized:
        store.lazy_init()
    store.assert_empty()


def test_assert_empty_reports_pending_cache(store):
    weight, events = {"grad": 0}, []
    pre_func, func = _gradient_callbacks(weight, 3, "pending", events)
    with store.set_split_bw(True):
        store.put(weight, pre_func, func)
    with pytest.raises(AssertionError, match="weight grad cache is not empty"):
        store.assert_empty()


@pytest.mark.parametrize("chunk", [0, 1])
def test_assert_empty_reports_pending_queue_in_any_chunk(store, monkeypatch, chunk):
    monkeypatch.setattr(zbpp_utils.parallel_state, "get_virtual_pipeline_model_parallel_world_size", lambda: 2)
    weight, events = {"grad": 0}, []
    pre_func, func = _gradient_callbacks(weight, 3, "pending", events)
    with store.set_split_bw(True):
        store.put(weight, pre_func, func)
        store.flush(chunk=chunk)
    assert store.cache == []
    with pytest.raises(AssertionError, match="weight grad queue is not empty"):
        store.assert_empty()
