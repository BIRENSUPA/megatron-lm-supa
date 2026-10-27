"""CPU boundary tests and real-device autograd, fusion, and TP communication tests.

The original CPU cases use reference boundaries; device cases below use real kernels and process groups.
"""

import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from megatron_supa.core import zbpp_utils
from megatron_supa.core.tensor_parallel import layers

pytestmark = [pytest.mark.sanity, pytest.mark.regression]


@pytest.fixture
def store(monkeypatch):
    store = zbpp_utils.WeightGradStore
    monkeypatch.setattr(store, "should_split_bw", False)
    monkeypatch.setattr(store, "cache", [])
    monkeypatch.setattr(store, "weight_grad_queue", None)
    args = SimpleNamespace(
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=None,
        overlap_grad_reduce=False,
        gradient_accumulation_fusion=True,
        transformer_impl="local",
    )
    monkeypatch.setattr(zbpp_utils, "get_args", lambda: args)
    monkeypatch.setattr(
        zbpp_utils,
        "parallel_state",
        SimpleNamespace(
            get_virtual_pipeline_model_parallel_world_size=lambda: None,
            get_pipeline_model_parallel_rank=lambda: 0,
        ),
    )
    return store


@pytest.fixture
def fusion(monkeypatch):
    calls, events = [], []

    def accumulate(route, total_input, grad_output, main_grad):
        assert total_input.device.type == grad_output.device.type == main_grad.device.type == "cpu"
        calls.append((route, total_input.detach().clone(), grad_output.detach().clone(), main_grad))
        events.append(("wgrad", route))
        with torch.no_grad():
            main_grad.add_(grad_output.t().matmul(total_input))

    monkeypatch.setattr(
        layers,
        "fused_weight_gradient_mlp_cuda",
        SimpleNamespace(
            wgrad_gemm_accum_fp32=lambda *args: accumulate("fp32", *args),
            wgrad_gemm_accum_fp16=lambda *args: accumulate("fp16", *args),
        ),
        raising=False,
    )
    return SimpleNamespace(calls=calls, events=events)


def _input_and_grad(microbatch=0):
    inputs = (
        [[1.0, -2.0, 0.5], [3.0, 0.25, -1.0]],
        [[0.25, -1.0, 2.0], [-2.0, 1.5, 0.5]],
    )
    gradients = (
        [[2.0, -1.0], [0.5, 3.0]],
        [[1.0, -0.5], [2.0, 1.5]],
    )
    return (
        torch.tensor(inputs[microbatch], device="cpu", requires_grad=True),
        torch.tensor(gradients[microbatch], device="cpu"),
    )


def _weight(dtype=torch.float32):
    weight = torch.tensor([[0.5, 1.0, -0.5], [-1.0, 0.25, 2.0]], device="cpu", requires_grad=True)
    weight.main_grad = torch.full(weight.shape, 0.25, dtype=dtype, device="cpu")
    return weight


def _linear(input, weight, bias=None, *, sequence_parallel=False, allreduce_dgrad=False, buffer=None, group=None):
    return layers.LinearWithGradAccumulationAndAsyncCommunication.apply(
        input, weight, bias, True, allreduce_dgrad, sequence_parallel, buffer, 0, group, 1
    )


class _TorchProxy:
    """Keep device and collective replacements local to the module under test."""

    def __init__(self, all_reduce=None):
        self.cuda = SimpleNamespace(current_device=lambda: torch.device("cpu"))
        if all_reduce is not None:
            self.distributed = SimpleNamespace(all_reduce=all_reduce)

    def __getattr__(self, name):
        return getattr(torch, name)


class _Handle:
    def __init__(self, name, events, complete):
        self.name, self.events, self.complete = name, events, complete
        self.waits = 0

    def wait(self):
        self.waits += 1
        assert self.waits == 1
        self.events.append(("wait", self.name))
        self.complete()


@pytest.mark.parametrize("split", [False, True], ids=["immediate", "split"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["fp32", "fp16", "bf16"])
def test_real_autograd_and_two_microbatch_fusion_accumulation(store, fusion, split, dtype):
    weight = _weight(dtype)
    expected = weight.main_grad.clone()
    route = "fp32" if dtype == torch.float32 else "fp16"
    tolerance = {torch.float32: 1e-6, torch.float16: 1e-3, torch.bfloat16: 1e-2}[dtype]

    with store.set_split_bw(split):
        for microbatch in range(2):
            input, grad_output = _input_and_grad(microbatch)
            bias = torch.tensor([0.75, -0.25], device="cpu", requires_grad=True)
            output = _linear(input, weight, bias)
            torch.testing.assert_close(output, input.detach().matmul(weight.detach().t()) + bias.detach())
            output.backward(grad_output)
            torch.testing.assert_close(input.grad, grad_output.matmul(weight.detach()))
            torch.testing.assert_close(bias.grad, grad_output.sum(dim=0))
            assert weight.grad is None

            if split:
                assert len(store.cache) == 1
                assert len(fusion.calls) == microbatch
                torch.testing.assert_close(weight.main_grad, expected)
                store.flush()
                assert store.cache == []
                assert store.weight_grad_queue[0][0].qsize() == 1
                assert len(fusion.calls) == microbatch
                torch.testing.assert_close(weight.main_grad, expected)
                store.pop()

            expected.add_(grad_output.t().matmul(input.detach()))
            torch.testing.assert_close(weight.main_grad, expected, rtol=tolerance, atol=tolerance)
            assert len(fusion.calls) == microbatch + 1
            actual_route, actual_input, actual_grad, target = fusion.calls[-1]
            assert actual_route == route and target is weight.main_grad
            torch.testing.assert_close(actual_input, input.detach())
            torch.testing.assert_close(actual_grad, grad_output)
            store.assert_empty()


def test_split_leaves_external_grad_output_buffer_untouched(store, fusion):
    input, grad_output = _input_and_grad()
    weight = _weight()
    initial = weight.main_grad.clone()
    sentinel = torch.tensor([42.0], device="cpu")
    buffer = [sentinel]
    with store.set_split_bw(True):
        output = _linear(input, weight, buffer=buffer)
        torch.testing.assert_close(output, input.detach().matmul(weight.detach().t()))
        output.backward(grad_output)
        torch.testing.assert_close(input.grad, grad_output.matmul(weight.detach()))
        assert len(buffer) == 1 and buffer[0] is sentinel
        torch.testing.assert_close(weight.main_grad, initial)
        assert fusion.calls == []
        store.flush()
        assert len(buffer) == 1 and buffer[0] is sentinel
        store.pop()
    assert len(buffer) == 1 and buffer[0] is sentinel
    torch.testing.assert_close(weight.main_grad, initial + grad_output.t().matmul(input.detach()))
    store.assert_empty()


def test_split_rejects_megatron_fsdp_weights(store, fusion):
    input, grad_output = _input_and_grad()
    weight = _weight()
    weight.__fsdp_param__ = object()
    with store.set_split_bw(True):
        with pytest.raises(RuntimeError, match="Zero Bubble split wgrad does not support Megatron-FSDP params"):
            _linear(input, weight).backward(grad_output)
    assert fusion.calls == []
    store.assert_empty()


@pytest.mark.parametrize("split", [False, True])
def test_rejects_unsupported_main_grad_dtype(store, fusion, split):
    input, grad_output = _input_and_grad()
    weight = _weight(torch.float64)
    with store.set_split_bw(split):
        with pytest.raises(RuntimeError, match="Unsupported gradient type for gradient accumulation fusion"):
            _linear(input, weight).backward(grad_output)
    assert fusion.calls == []
    store.assert_empty()


@pytest.mark.parametrize("split", [False, True], ids=["immediate", "split"])
def test_sequence_parallel_gathers_when_wgrad_runs_and_waits_for_dgrad(store, fusion, monkeypatch, split):
    input, _ = _input_and_grad()
    remote_input, _ = _input_and_grad(1)
    gathered_input = torch.cat((input.detach(), remote_input.detach()))
    grad_output = torch.tensor([[2.0, -1.0], [0.5, 3.0], [1.0, -0.5], [2.0, 1.5]], device="cpu")
    weight = _weight()
    initial = weight.main_grad.clone()
    group = SimpleNamespace(size=lambda: 2)
    handles, memory_requests = [], []
    peer_grad = torch.full_like(gathered_input, 0.5, device="cpu")

    def get_tensor(shape, dtype, name):
        memory_requests.append((shape, dtype, name))
        return torch.empty(shape, dtype=dtype, device="cpu")

    def gather(output, local_input, *, group, async_op=False):
        assert group is tp_group
        assert local_input is input
        assert list(output.shape) == [4, 3]
        fusion.events.append(("gather", async_op))

        def complete():
            return output.copy_(gathered_input)

        if async_op:
            handle = _Handle("gather", fusion.events, complete)
            handles.append(handle)
            return handle
        complete()
        return None

    def reduce_scatter(output, full_grad, *, group, async_op):
        assert group is tp_group and async_op is True
        assert output.device.type == "cpu" and output.shape == input.shape
        torch.testing.assert_close(full_grad, grad_output.matmul(weight.detach()))
        fusion.events.append(("reduce_scatter", async_op))
        handle = _Handle("reduce_scatter", fusion.events, lambda: output.copy_((full_grad + peer_grad).chunk(2)[0]))
        handles.append(handle)
        return handle

    tp_group = group
    monkeypatch.setattr(layers, "torch", _TorchProxy())
    monkeypatch.setattr(layers, "get_global_memory_buffer", lambda: SimpleNamespace(get_tensor=get_tensor))
    monkeypatch.setattr(layers, "dist_all_gather_func", gather)
    monkeypatch.setattr(layers, "dist_reduce_scatter_func", reduce_scatter)
    with store.set_split_bw(split):
        output = _linear(input, weight, sequence_parallel=True, group=group)
        torch.testing.assert_close(output, gathered_input.matmul(weight.detach().t()))
        assert fusion.events == [("gather", False)]
        output.backward(grad_output)
        torch.testing.assert_close(input.grad, (grad_output.matmul(weight.detach()) + peer_grad).chunk(2)[0])
        if split:
            backward_events = [("gather", False), ("reduce_scatter", True), ("wait", "reduce_scatter")]
            assert fusion.events == backward_events
            torch.testing.assert_close(weight.main_grad, initial)
            store.flush()
            assert fusion.events == backward_events
            torch.testing.assert_close(weight.main_grad, initial)
            store.pop()
            assert fusion.events == backward_events + [("gather", False), ("wgrad", "fp32")]
        else:
            assert fusion.events == [
                ("gather", False),
                ("gather", True),
                ("wait", "gather"),
                ("reduce_scatter", True),
                ("wgrad", "fp32"),
                ("wait", "reduce_scatter"),
            ]
    assert memory_requests == [([4, 3], input.dtype, "mpu")] * 2
    assert all(handle.waits == 1 for handle in handles)
    torch.testing.assert_close(weight.main_grad, initial + grad_output.t().matmul(gathered_input))
    store.assert_empty()


@pytest.mark.parametrize("split", [False, True], ids=["immediate", "split"])
def test_allreduce_dgrad_wait_completes_before_backward_returns(store, fusion, monkeypatch, split):
    input, grad_output = _input_and_grad()
    weight = _weight()
    initial = weight.main_grad.clone()
    tp_group, handles = object(), []
    peer_grad = torch.full_like(input, 0.5, device="cpu")

    def all_reduce(gradient, *, group, async_op):
        assert group is tp_group and async_op is True
        torch.testing.assert_close(gradient, grad_output.matmul(weight.detach()))
        fusion.events.append(("all_reduce", async_op))
        handle = _Handle("all_reduce", fusion.events, lambda: gradient.add_(peer_grad))
        handles.append(handle)
        return handle

    monkeypatch.setattr(layers, "torch", _TorchProxy(all_reduce=all_reduce))
    with store.set_split_bw(split):
        _linear(input, weight, allreduce_dgrad=True, group=tp_group).backward(grad_output)
        torch.testing.assert_close(input.grad, grad_output.matmul(weight.detach()) + peer_grad)
        if split:
            assert fusion.events == [("all_reduce", True), ("wait", "all_reduce")]
            torch.testing.assert_close(weight.main_grad, initial)
            store.flush()
            torch.testing.assert_close(weight.main_grad, initial)
            store.pop()
            assert fusion.events == [("all_reduce", True), ("wait", "all_reduce"), ("wgrad", "fp32")]
        else:
            assert fusion.events == [("all_reduce", True), ("wgrad", "fp32"), ("wait", "all_reduce")]
    assert len(handles) == 1 and handles[0].waits == 1
    torch.testing.assert_close(weight.main_grad, initial + grad_output.t().matmul(input.detach()))
    store.assert_empty()


@pytest.mark.parametrize("have_te", [False, True], ids=["torch", "te"])
@pytest.mark.parametrize("zero", [False, True], ids=["dummy", "zero"])
def test_split_dispatches_dummy_gradient_and_sets_main_grad_flag(store, fusion, monkeypatch, have_te, zero):
    input, grad_output = _input_and_grad()
    weight = _weight()
    initial = weight.main_grad.clone()
    weight.grad_added_to_main_grad = False
    weight.zero_out_wgrad = zero
    calls = []
    proxy = _TorchProxy()

    def allocate(name, shape, *, dtype, device, requires_grad):
        assert device == torch.device("cpu") and requires_grad is False
        calls.append((name, tuple(shape), dtype))
        return torch.full(shape, 0.0 if name == "zeros" else 0.125, dtype=dtype, device="cpu")

    def dummy(shape, dtype, **kwargs):
        assert kwargs == ({"zero": True} if zero else {})
        calls.append(("te", tuple(shape), dtype))
        return torch.full(shape, 0.0 if kwargs.get("zero", False) else 0.125, dtype=dtype, device="cpu")

    proxy.empty = lambda *args, **kwargs: allocate("empty", *args, **kwargs)
    proxy.zeros = lambda *args, **kwargs: allocate("zeros", *args, **kwargs)
    monkeypatch.setattr(layers, "torch", proxy)
    monkeypatch.setattr(layers, "HAVE_TE", have_te)
    monkeypatch.setattr(layers, "get_dummy_wgrad", dummy, raising=False)
    with store.set_split_bw(True):
        _linear(input, weight).backward(grad_output)
        assert weight.grad_added_to_main_grad is True
        route = "te" if have_te else "zeros" if zero else "empty"
        assert calls == [(route, tuple(weight.shape), input.dtype)]
        torch.testing.assert_close(weight.grad, torch.full_like(weight, 0.0 if zero else 0.125, device="cpu"))
        torch.testing.assert_close(weight.main_grad, initial)
        assert fusion.calls == []
        store.flush()
        store.pop()
    torch.testing.assert_close(weight.main_grad, initial + grad_output.t().matmul(input.detach()))
    store.assert_empty()


@contextmanager
def _device_rank_checks(device):
    """Make validation failures symmetric before either rank starts the next collective."""
    error = None
    try:
        yield
    except Exception as exc:
        error = exc
    failed = torch.tensor([int(error is not None)], dtype=torch.int32, device=device)
    torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
    if failed.item():
        messages = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(messages, None if error is None else repr(error))
        details = "; ".join(f"rank {rank}: {message}" for rank, message in enumerate(messages) if message)
        raise AssertionError(f"Real-device ZB validation failed: {details}") from error


@pytest.fixture
def device_case(request):
    """Use real TP/PP groups and restore the actual global args and WeightGradStore state."""
    from megatron.core import parallel_state
    from megatron.training import global_vars

    from tests.unit_tests.test_utilities import Utils

    tp_size, requires_fusion = request.param
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    store = zbpp_utils.WeightGradStore
    previous_args = global_vars._GLOBAL_ARGS
    previous_store = (store.should_split_bw, store.cache, store.weight_grad_queue)
    try:
        torch.cuda.set_device(local_rank)
        Utils.initialize_distributed()
        with _device_rank_checks(device):
            assert torch.distributed.get_world_size() == 2, "Real-device ZB layer tests require two ranks"
            assert os.environ.get("BR_UMD_BACKEND", "").lower() != "simulator", "Numerical UT requires real kernels"
            if requires_fusion:
                extension = getattr(layers, "fused_weight_gradient_mlp_cuda", None)
                assert layers._grad_accum_fusion_available and extension is not None, (
                    "Real-device ZB fusion dependency unavailable: fused_weight_gradient_mlp_cuda; "
                    f"layers._grad_accum_fusion_available={layers._grad_accum_fusion_available}"
                )
                assert callable(getattr(extension, "wgrad_gemm_accum_fp32", None)), "Missing real fp32 fusion entry"
                assert callable(getattr(extension, "wgrad_gemm_accum_fp16", None)), "Missing real fp16 fusion entry"

        # The dependency check above is collective and runs before constructing any TP/PP model groups.
        pp_size = 2 // tp_size
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size, pipeline_model_parallel_size=pp_size)
        args = SimpleNamespace(
            enable_zero_bubble=requires_fusion and pp_size > 1,
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            virtual_pipeline_model_parallel_size=None,
            overlap_grad_reduce=False,
            gradient_accumulation_fusion=requires_fusion,
            transformer_impl="local",
        )
        global_vars._GLOBAL_ARGS = args
        store.should_split_bw = False
        store.cache = []
        store.weight_grad_queue = None
        group = parallel_state.get_tensor_model_parallel_group()
        with _device_rank_checks(device):
            assert group.size() == tp_size
            assert parallel_state.get_pipeline_model_parallel_world_size() == pp_size
            assert zbpp_utils.get_args() is args
        yield SimpleNamespace(
            device=device,
            group=group,
            tp_rank=parallel_state.get_tensor_model_parallel_rank(),
            rank=torch.distributed.get_rank(),
            store=store,
        )
    finally:
        try:
            if torch.distributed.is_initialized():
                torch.cuda.synchronize(device)
                torch.distributed.barrier()
        finally:
            global_vars._GLOBAL_ARGS = previous_args
            store.should_split_bw, store.cache, store.weight_grad_queue = previous_store
            Utils.destroy_model_parallel()


def _device_cpu_input(rank, microbatch, dtype):
    values = torch.arange(32, dtype=torch.float32, device="cpu").reshape(4, 8)
    return (((values + 3 * microbatch + 2 * rank) % 13 - 6) / 8).to(dtype)


def _device_cpu_weight_and_bias(rank, dtype):
    values = torch.arange(64, dtype=torch.float32, device="cpu").reshape(8, 8)
    weight = (((values + 2 * rank) % 11 - 5) / 16).to(dtype)
    bias = ((torch.arange(8, dtype=torch.float32, device="cpu") - 3 + rank) / 16).to(dtype)
    return weight, bias


def _device_cpu_output_grad(rank, microbatch, rows, dtype):
    values = torch.arange(rows * 8, dtype=torch.float32, device="cpu").reshape(rows, 8)
    return (((values + 5 * microbatch + 3 * rank) % 9 - 4) / 16).to(dtype)


def _device_cpu_reference(input, weight, bias, grad_output):
    """Independent full-precision CPU autograd reference using quantized device operands."""
    input = input.double().requires_grad_(True)
    weight = weight.double().requires_grad_(True)
    bias = bias.double().requires_grad_(True)
    output = input.matmul(weight.t()) + bias
    output.backward(grad_output.double())
    return output.detach(), input.grad, weight.grad, bias.grad


def _assert_device_matches(actual, expected, dtype):
    assert actual.device.type != "cpu", "The result must come from a real device tensor"
    actual_cpu = actual.detach().to(device="cpu", dtype=torch.float32)
    assert torch.count_nonzero(actual_cpu).item() > 0, "A zero-only simulator result is not numerical validation"
    tolerance = {torch.float32: 2e-5, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(actual_cpu, expected.float(), rtol=tolerance, atol=tolerance)


def _device_linear(input, weight, bias, case, *, fused, mode="local"):
    return layers.LinearWithGradAccumulationAndAsyncCommunication.apply(
        input, weight, bias, fused, mode == "allreduce", mode == "sequence_parallel", None, 0, case.group, 1
    )


@pytest.mark.parametrize("device_case", [(1, True)], indirect=True, ids=["tp1-pp2"])
@pytest.mark.parametrize("split", [False, True], ids=["immediate", "split"])
@pytest.mark.parametrize(
    "activation_dtype,main_grad_dtype",
    [
        pytest.param(
            torch.float16,
            torch.float32,
            marks=pytest.mark.skip(
                reason=(
                    "Temporary: SUPA fused wgrad FP16->FP32 with width 8 produces incorrect main_grad; "
                    "reproduced with torch-supa-ext 0.0.1 and 2.0.0.dev0. Remove after the kernel fix."
                )
            ),
        ),
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
    ],
    ids=["fp16-fp32", "bf16-fp32", "fp16-fp16", "bf16-bf16"],
)
def test_device_fusion_two_microbatches_match_cpu_autograd(device_case, split, activation_dtype, main_grad_dtype):
    """Use supported activation/accumulator pairs with the actual fusion extension."""
    case = device_case
    cpu_weight, cpu_bias = _device_cpu_weight_and_bias(case.rank, activation_dtype)
    weight = cpu_weight.to(case.device).requires_grad_(True)
    expected_main_grad = torch.full(cpu_weight.shape, 0.25, dtype=main_grad_dtype, device="cpu")
    weight.main_grad = expected_main_grad.to(case.device)

    with case.store.set_split_bw(split):
        with _device_rank_checks(case.device):
            assert case.store.split_bw() is split
        for microbatch in range(2):
            cpu_input = _device_cpu_input(case.rank, microbatch, activation_dtype)
            cpu_grad = _device_cpu_output_grad(case.rank, microbatch, 4, activation_dtype)
            reference = _device_cpu_reference(cpu_input, cpu_weight, cpu_bias, cpu_grad)
            input = cpu_input.to(case.device).requires_grad_(True)
            bias = cpu_bias.to(case.device).requires_grad_(True)
            with _device_rank_checks(case.device):
                output = _device_linear(input, weight, bias, case, fused=True)
                _assert_device_matches(output, reference[0], activation_dtype)
                output.backward(cpu_grad.to(case.device))
                _assert_device_matches(input.grad, reference[1], activation_dtype)
                _assert_device_matches(bias.grad, reference[3], activation_dtype)
                assert weight.grad is None
                if split:
                    assert len(case.store.cache) == 1
                    torch.testing.assert_close(weight.main_grad.detach().cpu(), expected_main_grad, rtol=0, atol=0)
                    case.store.flush()
                    assert case.store.cache == []
                    assert case.store.weight_grad_queue[0][0].qsize() == 1
                    torch.testing.assert_close(weight.main_grad.detach().cpu(), expected_main_grad, rtol=0, atol=0)
                    case.store.pop()

                expected_main_grad = (expected_main_grad.double() + reference[2]).to(main_grad_dtype)
                _assert_device_matches(weight.main_grad, expected_main_grad, main_grad_dtype)
                case.store.assert_empty()
    with _device_rank_checks(case.device):
        assert case.store.should_split_bw is False


def _run_device_tp2_case(case, mode, dtype, fused, main_grad_dtype=None):
    rank = case.tp_rank
    accumulation_dtype = main_grad_dtype if fused and main_grad_dtype is not None else dtype
    cpu_weights_and_biases = [_device_cpu_weight_and_bias(peer, dtype) for peer in range(2)]
    cpu_weight, cpu_bias = cpu_weights_and_biases[rank]
    weight = cpu_weight.to(case.device).requires_grad_(True)
    expected_wgrad = torch.full(cpu_weight.shape, 0.25 if fused else 0.0, dtype=accumulation_dtype, device="cpu")
    if fused:
        weight.main_grad = expected_wgrad.to(case.device)

    for microbatch in range(2):
        if mode == "sequence_parallel":
            shards = [_device_cpu_input(peer, microbatch, dtype) for peer in range(2)]
            cpu_input = shards[rank]
            total_input = torch.cat(shards)
        else:
            # TP column shards share the input but use different local weights and output gradients.
            cpu_input = _device_cpu_input(0, microbatch, dtype)
            total_input = cpu_input
        gradients = [_device_cpu_output_grad(peer, microbatch, total_input.shape[0], dtype) for peer in range(2)]
        references = [
            _device_cpu_reference(total_input, peer_weight, peer_bias, gradients[peer])
            for peer, (peer_weight, peer_bias) in enumerate(cpu_weights_and_biases)
        ]
        reference_dgrad = references[0][1] + references[1][1]
        if mode == "sequence_parallel":
            reference_dgrad = reference_dgrad.chunk(2)[rank]
        contribution = references[rank][2]
        if not fused:
            contribution = contribution.to(dtype)
        expected_wgrad = (expected_wgrad.double() + contribution.double()).to(accumulation_dtype)
        input = cpu_input.to(case.device).requires_grad_(True)
        bias = cpu_bias.to(case.device).requires_grad_(True)

        # Complete every rank's real forward/backward collectives before entering the assertion gate.
        output = _device_linear(input, weight, bias, case, fused=fused, mode=mode)
        output.backward(gradients[rank].to(case.device))
        with _device_rank_checks(case.device):
            _assert_device_matches(output, references[rank][0], dtype)
            _assert_device_matches(input.grad, reference_dgrad, dtype)
            _assert_device_matches(bias.grad, references[rank][3], dtype)
            _assert_device_matches(weight.main_grad if fused else weight.grad, expected_wgrad, accumulation_dtype)
            if fused:
                assert weight.grad is None
            assert case.store.should_split_bw is False
            case.store.assert_empty()


@pytest.mark.parametrize("device_case", [(2, False)], indirect=True, ids=["tp2-pp1"])
@pytest.mark.parametrize("mode", ["allreduce", "sequence_parallel"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["fp32", "fp16", "bf16"])
def test_device_tp2_without_fusion_matches_cpu_collective_reference(device_case, mode, dtype):
    """Run true all-reduce or all-gather/reduce-scatter independently of the fusion extension."""
    _run_device_tp2_case(device_case, mode, dtype, fused=False)


@pytest.mark.parametrize("device_case", [(2, True)], indirect=True, ids=["tp2-pp1"])
@pytest.mark.parametrize("mode", ["allreduce", "sequence_parallel"])
def test_device_tp2_immediate_fusion_matches_cpu_collective_reference(device_case, mode):
    """Real TP2 fusion is immediate; topology-consistent TP2/PP2 split testing needs four ranks."""
    _run_device_tp2_case(device_case, mode, torch.bfloat16, fused=True, main_grad_dtype=torch.float32)
