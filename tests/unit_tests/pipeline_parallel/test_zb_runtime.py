"""Check ZB orchestration with boundary doubles and a separate real-device PP2 numerical test."""

import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from megatron.core import zbpp_utils
from megatron.core.pipeline_parallel.zerobubble import runtime
from megatron.core.pipeline_parallel.zerobubble.scheduler import zb
from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import FuncType
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import CudaGraphScope

pytestmark = [pytest.mark.sanity, pytest.mark.regression]


class _RuntimeCase:
    """Keep the executor, scheduler and weight-gradient store real; replace their external boundaries."""

    def __init__(self, monkeypatch, stage=0, initial_split=False):
        self.stage = stage
        self.initial_split = initial_split
        self.events = []
        self.group_calls = []
        self.communicator_builds = []
        self.shape_calls = []
        self.p2p_calls = []
        self.finalize_calls = []
        self.timer_calls = []
        self.grad_sync_parameters = []
        self.backward_sync_active = {}
        self.sync_active = False
        self.forward_data_store = None
        self.store = zbpp_utils.WeightGradStore
        self.config = SimpleNamespace(
            overlap_p2p_comm=False,
            no_sync_func=self.no_sync,
            grad_sync_func=None,
            finalize_model_grads_func=self.finalize,
            calculate_per_token_loss=True,
            deallocate_pipeline_outputs=True,
            timers=None,
            barrier_with_L1_time=False,
        )
        self.model = torch.nn.Module()
        self.model.register_parameter("weight", torch.nn.Parameter(torch.tensor([2.0], device="cpu")))
        self.model.weight.grad = torch.zeros(1, device="cpu")
        self.model.config = self.config
        self.data_iterator = iter(("microbatch-0", "microbatch-1"))
        self.forward_step_func = lambda: None
        self.groups = {name: object() for name in ("tp", "pp", "embd", "pos_embd", "dp_cp", "tp_dp_cp")}
        self.groups["cp"] = SimpleNamespace(size=lambda: 3)
        self.pg_collection = ProcessGroupCollection(**self.groups)
        self.recv_shapes = [(8, 2, 4)]
        self.send_shapes = [(6, 2, 4)]
        self.embedding = object()
        self.inputs = [SimpleNamespace(kind="input", microbatch=m) for m in range(2)]
        self.outputs = [SimpleNamespace(kind="output", microbatch=m) for m in range(2)]
        self.output_grads = [SimpleNamespace(kind="output-grad", microbatch=m) for m in range(2)]
        self.input_grads = [SimpleNamespace(kind="input-grad", microbatch=m) for m in range(2)]
        self.losses = [{"microbatch": m, "loss": m + 0.5} for m in range(2)]
        self.communicator = SimpleNamespace(
            total_stages=2,
            current_stage=stage,
            is_pp_first_stage=stage == 0,
            is_pp_last_stage=stage == 1,
            recv_forward=self.recv_forward,
            send_forward=self.send_forward,
            recv_backward=self.recv_backward,
            send_backward=self.send_backward,
        )

        monkeypatch.setattr(zb, "_SCHEDULE_CACHE", {})
        monkeypatch.setattr(self.store, "should_split_bw", initial_split)
        monkeypatch.setattr(self.store, "cache", [])
        monkeypatch.setattr(self.store, "weight_grad_queue", None)
        supported_args = SimpleNamespace(
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=None,
            overlap_grad_reduce=False,
            gradient_accumulation_fusion=True,
            transformer_impl="local",
        )
        monkeypatch.setattr(zbpp_utils, "get_args", lambda: supported_args)
        monkeypatch.setattr(
            zbpp_utils,
            "parallel_state",
            SimpleNamespace(
                get_virtual_pipeline_model_parallel_world_size=lambda: None,
                get_pipeline_model_parallel_rank=lambda: stage,
            ),
        )
        monkeypatch.setattr(runtime, "torch", SimpleNamespace(zeros=self.allocate_tokens, int=torch.int))
        monkeypatch.setattr(runtime, "P2PCommunicator", self.build_communicator)
        monkeypatch.setattr(
            runtime,
            "parallel_state",
            SimpleNamespace(
                get_pipeline_model_parallel_group=lambda: self.group("pp"),
                get_tensor_model_parallel_group=lambda: self.group("tp"),
                get_context_parallel_group=lambda: self.group("cp"),
                get_embedding_group=lambda **kw: self.group("embd", **kw),
                get_position_embedding_group=lambda **kw: self.group("pos_embd", **kw),
                get_data_parallel_group=lambda **kw: self.group("dp_cp", **kw),
                get_tensor_and_data_parallel_group=lambda **kw: self.group("tp_dp_cp", **kw),
            ),
        )
        for name in (
            "get_tensor_shapes",
            "forward_step",
            "backward_step",
            "deallocate_output_tensor",
            "clear_embedding_activation_buffer",
            "finish_embedding_wgrad_compute",
            "create_cudagraphs",
        ):
            monkeypatch.setattr(runtime, name, getattr(self, name))

        # The scheduler has separate tests. Here its real output is the executor's input contract.
        self.schedule = zb.get_zb_schedule(2, 2, stage)
        self.forward_receives = iter(node.microbatch for node in self.schedule if node.type == FuncType.RECV_FORWARD)
        self.backward_receives = iter(node.microbatch for node in self.schedule if node.type == FuncType.RECV_BACKWARD)

    def run(self, **overrides):
        options = dict(
            forward_step_func=self.forward_step_func,
            data_iterator=self.data_iterator,
            model=self.model,
            num_microbatches=2,
            seq_length=8,
            micro_batch_size=2,
            decoder_seq_length=6,
            collect_non_loss_data=True,
            first_val_step=False,
            p2p_communicator=self.communicator,
            pg_collection=self.pg_collection,
            force_all_reduce=True,
        )
        options.update(overrides)
        return runtime.forward_backward_zero_bubble(**options)

    def group(self, name, **kwargs):
        self.group_calls.append((name, kwargs))
        return self.groups[name]

    def build_communicator(self, *, pp_group, config):
        self.communicator_builds.append((pp_group, config))
        return self.communicator

    def get_tensor_shapes(self, **kwargs):
        self.shape_calls.append(kwargs)
        return self.recv_shapes if len(self.shape_calls) == 1 else self.send_shapes

    def allocate_tokens(self, shape, *, dtype, device):
        assert shape == []
        assert dtype == torch.int
        assert device == "cuda"
        self.events.append(("allocate_tokens", None))
        return torch.zeros(shape, dtype=dtype, device="cpu")

    @contextmanager
    def no_sync(self):
        self.events.append(("no_sync_enter", None))
        self.sync_active = True
        try:
            yield
        finally:
            self.sync_active = False
            self.events.append(("no_sync_exit", None))

    def grad_sync(self, parameters):
        self.grad_sync_parameters.append(list(parameters))
        self.events.append(("grad_sync", None))

    def recv_forward(self, shapes, is_first_stage):
        microbatch = next(self.forward_receives)
        self.p2p_calls.append(("RECV_FORWARD", microbatch, shapes, is_first_stage))
        self.events.append(("RECV_FORWARD", microbatch))
        return self.inputs[microbatch]

    def send_forward(self, output, is_last_stage):
        microbatch = output.microbatch
        assert output is self.outputs[microbatch]
        self.p2p_calls.append(("SEND_FORWARD", microbatch, output, is_last_stage))
        self.events.append(("SEND_FORWARD", microbatch))

    def recv_backward(self, shapes, is_last_stage):
        microbatch = next(self.backward_receives)
        self.p2p_calls.append(("RECV_BACKWARD", microbatch, shapes, is_last_stage))
        self.events.append(("RECV_BACKWARD", microbatch))
        return self.output_grads[microbatch]

    def send_backward(self, input_grad, is_first_stage):
        microbatch = input_grad.microbatch
        assert input_grad is self.input_grads[microbatch]
        assert self.store.cache == []
        assert not self.store.weight_grad_queue[0][0].empty()
        assert ("W", microbatch) not in self.events
        self.p2p_calls.append(("SEND_BACKWARD", microbatch, input_grad, is_first_stage))
        self.events.append(("SEND_BACKWARD", microbatch))

    def forward_step(self, step_func, iterator, model, count, input_tensor, data_store, config, cp_size, **kwargs):
        microbatch = kwargs["current_microbatch"]
        assert step_func is self.forward_step_func
        assert iterator is self.data_iterator
        assert model is self.model
        assert count == 2
        assert input_tensor is self.inputs[microbatch]
        assert config is self.config
        assert cp_size == 3
        assert kwargs == dict(
            collect_non_loss_data=True,
            checkpoint_activations_microbatch=None,
            is_first_microbatch=microbatch == 0,
            current_microbatch=microbatch,
            is_last_stage=self.stage == 1,
        )
        assert self.store.should_split_bw is self.initial_split
        if self.forward_data_store is None:
            self.forward_data_store = data_store
        assert data_store is self.forward_data_store
        data_store.append(self.losses[microbatch])
        self.events.append(("F", microbatch))
        return self.outputs[microbatch], (3, 5)[microbatch]

    def backward_step(self, input_tensor, output, output_grad, config):
        microbatch = output.microbatch
        assert input_tensor is self.inputs[microbatch]
        assert output is self.outputs[microbatch]
        assert output_grad is self.output_grads[microbatch]
        assert config is self.config
        assert self.store.should_split_bw is True
        assert self.store.split_bw() is True
        self.backward_sync_active[microbatch] = self.sync_active
        self.events.append(("B", microbatch))

        def prepare(*, async_op):
            assert async_op is False
            assert self.store.should_split_bw is self.initial_split
            assert self.store.cache == []
            self.events.append(("prepare_W", microbatch))
            return self.model.weight, microbatch + 1

        def weight_grad(weight, amount):
            assert weight is self.model.weight
            assert amount == microbatch + 1
            weight.grad.add_(amount)
            self.events.append(("W", microbatch))

        self.store.put(self.model.weight, prepare, weight_grad)
        assert ("prepare_W", microbatch) not in self.events
        assert ("W", microbatch) not in self.events
        return self.input_grads[microbatch]

    def deallocate_output_tensor(self, output, enabled):
        assert output is self.outputs[output.microbatch]
        assert enabled is self.config.deallocate_pipeline_outputs
        self.events.append(("deallocate", output.microbatch))

    def clear_embedding_activation_buffer(self, config, model, is_last_stage):
        assert config is self.config
        assert model is self.model
        assert is_last_stage is (self.stage == 1)
        self.events.append(("embedding_clear", None))
        return self.embedding

    def finish_embedding_wgrad_compute(self, config, embedding, is_last_stage, tp_group):
        assert config is self.config
        assert embedding is self.embedding
        assert is_last_stage is (self.stage == 1)
        assert tp_group is self.groups["tp"]
        self.events.append(("embedding_finish", None))

    def finalize(self, models, tokens, *, pg_collection, force_all_reduce):
        assert len(models) == 1 and models[0] is self.model
        self.store.assert_empty()
        self.finalize_calls.append((tokens, pg_collection, force_all_reduce))
        self.events.append(("finalize", None))

    def timers(self, name, **kwargs):
        self.timer_calls.append((name, kwargs))
        return SimpleNamespace(start=self.timer_start, stop=self.timer_stop)

    def timer_start(self, *, barrier):
        assert barrier is self.config.barrier_with_L1_time
        self.events.append(("timer_start", barrier))

    def timer_stop(self):
        self.events.append(("timer_stop", None))

    def create_cudagraphs(self):
        self.events.append(("cuda_graphs", None))

    def assert_completed(self, result):
        assert result is self.forward_data_store
        assert len(result) == 2
        assert all(actual is expected for actual, expected in zip(result, self.losses))
        executed = [event for event in self.events if event[0] in FuncType.__members__]
        assert executed == [(node.type.name, node.microbatch) for node in self.schedule]
        assert self.store.should_split_bw is self.initial_split
        assert self.store.cache == []
        assert self.store.weight_grad_queue is not None
        assert all(queue.empty() for chunk in self.store.weight_grad_queue for queue in chunk)
        assert torch.equal(self.model.weight.grad, torch.tensor([3.0], device="cpu"))
        assert self.sync_active is False
        assert (
            self.shape_calls
            == [
                dict(
                    seq_length=8,
                    micro_batch_size=2,
                    decoder_seq_length=6,
                    config=self.config,
                    tp_group=self.groups["tp"],
                    cp_group=self.groups["cp"],
                )
            ]
            * 2
        )
        for kind, microbatch, payload, boundary in self.p2p_calls:
            expected_payload = {
                "RECV_FORWARD": self.recv_shapes,
                "SEND_FORWARD": self.outputs[microbatch],
                "RECV_BACKWARD": self.send_shapes,
                "SEND_BACKWARD": self.input_grads[microbatch],
            }[kind]
            assert payload is expected_payload
            assert boundary is (self.stage == (0 if kind in ("RECV_FORWARD", "SEND_BACKWARD") else 1))
        for microbatch in range(2):

            def index(name):
                return self.events.index((name, microbatch))

            assert self.events.count(("deallocate", microbatch)) == 1
            assert index("RECV_FORWARD") < index("F") < index("SEND_FORWARD") < index("deallocate")
            assert index("RECV_BACKWARD") < index("B") < index("SEND_BACKWARD")
            assert index("B") < index("prepare_W") < index("W")


@pytest.fixture
def runtime_case(monkeypatch):
    return lambda **kwargs: _RuntimeCase(monkeypatch, **kwargs)


@pytest.mark.parametrize("stage", [0, 1])
@pytest.mark.parametrize("deallocate", [False, True])
def test_executes_real_schedule_with_single_element_lists(runtime_case, stage, deallocate):
    case = runtime_case(stage=stage)
    case.config.deallocate_pipeline_outputs = deallocate

    result = case.run(model=[case.model], data_iterator=[case.data_iterator])

    case.assert_completed(result)
    assert case.group_calls == []
    assert case.communicator_builds == []
    assert case.finalize_calls[0][1] is case.pg_collection
    assert case.finalize_calls[0][2] is True
    assert case.timer_calls == []


@pytest.mark.parametrize(
    "invalid,error,message",
    [
        ("model_chunks", AssertionError, "model chunking"),
        ("iterator_chunks", AssertionError, "model chunking"),
        ("forward_only", AssertionError, "forward_only"),
        ("adjust_shapes", AssertionError, "adjust_tensor_shapes_fn"),
        ("overlap", ValueError, "overlapping p2p communication"),
        ("communicator_only", ValueError, "Provide both"),
        ("groups_only", ValueError, "Provide both"),
    ],
)
def test_rejects_unsupported_entry_options(runtime_case, invalid, error, message):
    case = runtime_case()
    options = {
        "model_chunks": {"model": [case.model, case.model]},
        "iterator_chunks": {"data_iterator": [case.data_iterator, case.data_iterator]},
        "forward_only": {"forward_only": True},
        "adjust_shapes": {"adjust_tensor_shapes_fn": lambda shapes: shapes},
        "overlap": {},
        "communicator_only": {"pg_collection": None},
        "groups_only": {"p2p_communicator": None},
    }[invalid]
    if invalid == "overlap":
        case.config.overlap_p2p_comm = True

    with pytest.raises(error, match=message):
        case.run(**options)

    assert case.events == []
    assert case.shape_calls == []
    assert case.group_calls == []
    assert case.communicator_builds == []


def test_builds_communicator_and_real_group_collection_from_parallel_state(runtime_case):
    case = runtime_case(stage=1)

    result = case.run(p2p_communicator=None, pg_collection=None)

    case.assert_completed(result)
    assert len(case.communicator_builds) == 1
    assert case.communicator_builds[0][0] is case.groups["pp"]
    assert case.communicator_builds[0][1] is case.config
    actual_groups = case.finalize_calls[0][1]
    assert isinstance(actual_groups, ProcessGroupCollection)
    assert actual_groups is not case.pg_collection
    for name, expected in case.groups.items():
        assert getattr(actual_groups, name) is expected
    assert case.group_calls == [
        ("pp", {}),
        ("tp", {}),
        ("cp", {}),
        ("pp", {}),
        ("embd", {"check_initialized": False}),
        ("pos_embd", {"check_initialized": False}),
        ("dp_cp", {"with_context_parallel": True, "partial_data_parallel": False}),
        ("tp_dp_cp", {"with_context_parallel": True}),
    ]


@pytest.mark.parametrize("stage", [0, 1])
@pytest.mark.parametrize("custom_grad_sync", [False, True])
@pytest.mark.parametrize("use_no_sync", [False, True])
def test_gradient_sync_timing_and_parameter_identity(runtime_case, stage, custom_grad_sync, use_no_sync):
    case = runtime_case(stage=stage)
    case.config.no_sync_func = case.no_sync if use_no_sync else None
    case.config.grad_sync_func = case.grad_sync if custom_grad_sync else None

    result = case.run()

    case.assert_completed(result)
    enter, exit_ = ("no_sync_enter", None), ("no_sync_exit", None)
    if use_no_sync:
        assert case.events.count(enter) == case.events.count(exit_) == 1
        assert case.events.index(enter) < case.events.index(("F", 0))
        assert case.backward_sync_active[0] is True
        if stage == 0 or not custom_grad_sync:
            assert case.events.index(exit_) < case.events.index(("B", 1))
            assert case.backward_sync_active[1] is False
        else:
            assert case.events.index(exit_) > max(case.events.index(("W", m)) for m in range(2))
            assert case.backward_sync_active[1] is True
    else:
        assert enter not in case.events and exit_ not in case.events
        assert case.backward_sync_active == {0: False, 1: False}
    if custom_grad_sync and stage == 1:
        assert len(case.grad_sync_parameters) == 1
        parameters = list(case.model.parameters())
        assert len(case.grad_sync_parameters[0]) == len(parameters)
        assert all(actual is expected for actual, expected in zip(case.grad_sync_parameters[0], parameters))
        assert case.events.index(("grad_sync", None)) > max(case.events.index(("W", m)) for m in range(2))
        assert case.events.index(("grad_sync", None)) < case.events.index(("finalize", None))
        if use_no_sync:
            assert case.events.index(exit_) < case.events.index(("grad_sync", None))
    else:
        assert case.grad_sync_parameters == []


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("per_token_loss", [False, True])
@pytest.mark.parametrize("barrier", [False, True])
def test_finalization_tokens_embedding_order_and_timers(runtime_case, enabled, per_token_loss, barrier):
    case = runtime_case(stage=1)
    case.config.finalize_model_grads_func = case.finalize if enabled else None
    case.config.calculate_per_token_loss = per_token_loss
    case.config.timers = case.timers
    case.config.barrier_with_L1_time = barrier

    result = case.run(force_all_reduce=False)

    case.assert_completed(result)
    start, stop = ("timer_start", barrier), ("timer_stop", None)
    assert case.timer_calls == [("forward-backward", {"log_level": 1}), ("forward-backward", {})]
    assert case.events.count(start) == case.events.count(stop) == 1
    assert case.events.index(start) < case.events.index(("no_sync_enter", None))
    assert case.events.index(stop) > max(case.events.index(("W", m)) for m in range(2))
    if enabled:
        assert len(case.finalize_calls) == 1
        tokens, groups, force_all_reduce = case.finalize_calls[0]
        assert groups is case.pg_collection
        assert force_all_reduce is False
        if per_token_loss:
            assert tokens.device.type == "cpu"
            assert tokens.dtype == torch.int
            assert tokens.shape == torch.Size([])
            assert tokens.item() == 8
        else:
            assert tokens is None
        clear, finish, finalize = ("embedding_clear", None), ("embedding_finish", None), ("finalize", None)
        assert case.events.count(clear) == case.events.count(finish) == case.events.count(finalize) == 1
        assert case.events.index(clear) < case.events.index(start)
        assert case.events.index(finish) > max(case.events.index(("W", m)) for m in range(2))
        assert case.events.index(finish) < case.events.index(finalize) < case.events.index(stop)
    else:
        assert case.finalize_calls == []
        assert not any(name.startswith("embedding_") for name, _ in case.events)


@pytest.mark.parametrize(
    "mode,expected",
    [("unconfigured", False), ("nonlocal", False), ("full-iteration", False), ("local", True)],
)
def test_creates_cuda_graphs_only_for_local_partial_scope(runtime_case, mode, expected):
    case = runtime_case()
    case.config.timers = case.timers
    if mode != "unconfigured":
        case.config.cuda_graph_impl = "transformer_engine" if mode == "nonlocal" else "local"
        case.config.cuda_graph_scope = [CudaGraphScope.full_iteration] if mode == "full-iteration" else []

    result = case.run()

    case.assert_completed(result)
    assert case.events.count(("cuda_graphs", None)) == int(expected)
    if expected:
        assert case.events.index(("cuda_graphs", None)) > case.events.index(("timer_stop", None))


def test_preserves_preexisting_split_state_after_real_weight_callbacks(runtime_case):
    case = runtime_case(initial_split=True)

    result = case.run()

    case.assert_completed(result)
    assert case.store.should_split_bw is True


@contextmanager
def _device_runtime_rank_checks(device):
    """Report setup and numerical assertion failures to both ranks before their next collective."""
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
        raise AssertionError(f"Real-device ZB runtime validation failed: {details}") from error


@pytest.fixture
def device_runtime_case():
    """Use real TP1/PP2 groups, fusion and global arguments without the CPU boundary fixtures."""
    from megatron.core import parallel_state
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.training import global_vars

    from megatron_supa.core.tensor_parallel import layers
    from tests.unit_tests.test_utilities import Utils

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    store = zbpp_utils.WeightGradStore
    previous_args = global_vars._GLOBAL_ARGS
    previous_store = (store.should_split_bw, store.cache, store.weight_grad_queue)
    previous_schedule_cache = zb._SCHEDULE_CACHE
    try:
        torch.cuda.set_device(local_rank)
        print(f"[ZB PP2 rank {local_rank}] initializing distributed groups", flush=True)
        Utils.initialize_distributed()
        with _device_runtime_rank_checks(device):
            assert torch.distributed.get_world_size() == 2, "Real-device ZB runtime requires exactly two ranks"
            assert os.environ.get("BR_UMD_BACKEND", "").lower() != "simulator", "Numerical UT requires real kernels"
            extension = getattr(layers, "fused_weight_gradient_mlp_cuda", None)
            assert layers._grad_accum_fusion_available and extension is not None, (
                "Real-device ZB runtime fusion dependency unavailable: fused_weight_gradient_mlp_cuda; "
                f"layers._grad_accum_fusion_available={layers._grad_accum_fusion_available}"
            )
            assert callable(getattr(extension, "wgrad_gemm_accum_fp32", None)), "Missing real FP32 fusion accumulator"

        Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
        args = SimpleNamespace(
            enable_zero_bubble=True,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=None,
            overlap_grad_reduce=False,
            gradient_accumulation_fusion=True,
            transformer_impl="local",
        )
        global_vars._GLOBAL_ARGS = args
        store.should_split_bw = False
        store.cache = []
        store.weight_grad_queue = None
        zb._SCHEDULE_CACHE = {}
        with _device_runtime_rank_checks(device):
            config = TransformerConfig(
                num_layers=2,
                hidden_size=16,
                num_attention_heads=1,
                ffn_hidden_size=32,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=2,
                virtual_pipeline_model_parallel_size=None,
                params_dtype=torch.bfloat16,
                pipeline_dtype=torch.bfloat16,
                bf16=True,
                gradient_accumulation_fusion=True,
                sequence_parallel=False,
                overlap_p2p_comm=False,
                # Keep Core's real independent isend/irecv path, without batch_p2p_sync's device-wide sync.
                batch_p2p_comm=False,
                deallocate_pipeline_outputs=False,
                calculate_per_token_loss=False,
                enable_autocast=False,
                cuda_graph_impl="none",
            )
            group = parallel_state.get_tensor_model_parallel_group()
            stage = parallel_state.get_pipeline_model_parallel_rank()
            assert group.size() == 1
            assert parallel_state.get_pipeline_model_parallel_world_size() == 2
            assert parallel_state.get_context_parallel_group().size() == 1
            assert zbpp_utils.get_args() is args
            assert store.is_supported() is True

            # Cmodel's first BF16 GEMM/fusion launch can take longer than the P2P
            # rendezvous timeout. Warm both ranks before either enters runtime's recv.
            # These temporary tensors are independent of the model and its gradients.
            print(f"[ZB PP2 rank {local_rank}] TP1/PP2 groups ready; warming real kernels", flush=True)
            warmup_input = torch.full((2, 1, 16), 0.5, dtype=torch.bfloat16, device=device, requires_grad=True)
            warmup_weight = torch.nn.Parameter(torch.full((16, 16), 0.0625, dtype=torch.bfloat16, device=device))
            warmup_weight.main_grad = torch.zeros_like(warmup_weight, dtype=torch.float32)
            warmup_target = torch.full((2, 1, 16), 0.25, dtype=torch.bfloat16, device=device)
            assert store.should_split_bw is False
            store.assert_empty()
            warmup_output = layers.LinearWithGradAccumulationAndAsyncCommunication.apply(
                warmup_input, warmup_weight, None, True, False, False, None, 0, group, 1
            )
            print(f"[ZB PP2 rank {local_rank}] warmup forward launched; warming loss and backward", flush=True)
            warmup_loss = (warmup_output.float() - warmup_target.float()).square().mean()
            warmup_loss /= 2
            warmup_loss.backward()
            torch.cuda.synchronize(device)
            assert warmup_input.grad is not None
            assert store.should_split_bw is False
            store.assert_empty()
            assert store.weight_grad_queue is None
            del warmup_loss, warmup_output, warmup_target, warmup_weight, warmup_input
            print(f"[ZB PP2 rank {local_rank}] kernel warmup complete; synchronizing ranks", flush=True)
        # The rank-check all-reduce above completes only after both local warmups.
        print(f"[ZB PP2 rank {local_rank}] both ranks ready for the real runtime", flush=True)
        yield SimpleNamespace(device=device, group=group, stage=stage, config=config, store=store)
    finally:
        try:
            if torch.distributed.is_initialized():
                print(f"[ZB PP2 rank {local_rank}] synchronizing for fixture cleanup", flush=True)
                torch.cuda.synchronize(device)
                torch.distributed.barrier()
        finally:
            global_vars._GLOBAL_ARGS = previous_args
            store.should_split_bw, store.cache, store.weight_grad_queue = previous_store
            zb._SCHEDULE_CACHE = previous_schedule_cache
            Utils.destroy_model_parallel()


class _DevicePipelineLinear(torch.nn.Module):
    """A real bias-free linear stage implementing Core's pipeline model interface."""

    def __init__(self, case, cpu_weight):
        super().__init__()
        self.config = case.config
        self.group = case.group
        self.stage = case.stage
        self.weight = torch.nn.Parameter(cpu_weight.to(case.device))
        self.weight.main_grad = torch.zeros_like(self.weight, dtype=torch.float32)
        self.input_tensor = None
        self.forward_inputs = []
        self.forward_outputs = []

    def set_input_tensor(self, input_tensors):
        assert len(input_tensors) == 1
        self.input_tensor = input_tensors[0]

    def forward(self, data):
        from megatron_supa.core.tensor_parallel.layers import LinearWithGradAccumulationAndAsyncCommunication

        input_tensor = data if self.stage == 0 else self.input_tensor
        output = LinearWithGradAccumulationAndAsyncCommunication.apply(
            input_tensor, self.weight, None, True, False, False, None, 0, self.group, 1
        )
        self.forward_inputs.append(input_tensor.detach().clone())
        self.forward_outputs.append(output.detach().clone())
        return output


def _device_runtime_cpu_reference():
    """Independent two-layer CPU autograd with BF16 operands and explicit pipeline rounding."""
    dtype = torch.bfloat16
    weight_values = torch.arange(256, dtype=torch.float32, device="cpu").reshape(16, 16)
    weights = [
        ((weight_values % 17 - 8) / 32).to(dtype),
        ((weight_values * 3 % 19 - 9) / 32).to(dtype),
    ]
    values = torch.arange(32, dtype=torch.float32, device="cpu").reshape(2, 1, 16)
    inputs = [((values + 3 * microbatch) % 17 - 8).div(8).to(dtype) for microbatch in range(2)]
    targets = [((values * 3 + 5 * microbatch) % 19 - 9).div(16).to(dtype) for microbatch in range(2)]
    reference_weights = [weight.double().requires_grad_(True) for weight in weights]
    hidden_states, outputs, losses, contributions = [], [], [], []
    for input_tensor, target in zip(inputs, targets):
        # Casts round both activations and their reverse-mode gradients at each BF16 stage boundary.
        hidden = input_tensor.double().matmul(reference_weights[0].t()).to(dtype).double()
        output = hidden.matmul(reference_weights[1].t()).to(dtype).double()
        loss = (output.float() - target.float()).square().mean()
        hidden_states.append(hidden.detach())
        outputs.append(output.detach())
        losses.append(loss)
        gradients = torch.autograd.grad(loss / 2, reference_weights, retain_graph=True)
        contributions.append([gradient.detach() for gradient in gradients])
    # Core's two-item loss return is scaled by CP_size / num_microbatches, here 1 / 2.
    ((losses[0] + losses[1]) / 2).backward()
    return SimpleNamespace(
        weights=weights,
        inputs=inputs,
        targets=targets,
        hidden_states=hidden_states,
        outputs=outputs,
        losses=[loss.detach() for loss in losses],
        gradients=[weight.grad.detach() for weight in reference_weights],
        contributions=contributions,
    )


def _assert_device_runtime_close(actual, expected):
    assert actual.device.type != "cpu", "The runtime result must be produced on a real device"
    actual_cpu = actual.detach().to(device="cpu", dtype=torch.float32)
    assert torch.count_nonzero(actual_cpu).item() > 0, "Zero-only simulator output is not numerical validation"
    torch.testing.assert_close(actual_cpu, expected.float(), rtol=2e-2, atol=2e-4)
    return actual_cpu


def test_device_pp2_two_microbatches_match_cpu_two_layer_autograd(device_runtime_case):
    """Exercise real Core F/B, P2P, ZB scheduling and deferred native fusion on two pipeline ranks."""
    case = device_runtime_case
    with _device_runtime_rank_checks(case.device):
        reference = _device_runtime_cpu_reference()
        model = _DevicePipelineLinear(case, reference.weights[case.stage])
        data_iterator = iter(
            [
                (microbatch, input_tensor.to(case.device), target.to(case.device))
                for microbatch, (input_tensor, target) in enumerate(zip(reference.inputs, reference.targets))
            ]
        )
        expected_gradient = reference.gradients[case.stage].float()
        assert not torch.equal(reference.inputs[0], reference.inputs[1])
        assert not torch.allclose(reference.contributions[0][case.stage], reference.contributions[1][case.stage])
        assert torch.count_nonzero(expected_gradient).item() > 0
        for contribution in reference.contributions:
            assert torch.count_nonzero(contribution[case.stage]).item() > 0
            assert not torch.allclose(expected_gradient, contribution[case.stage].float(), rtol=2e-2, atol=2e-4)
        assert case.store.should_split_bw is False
        case.store.assert_empty()

    def forward_step(data_iterator, stage_model):
        microbatch, input_tensor, target = next(data_iterator)
        output = stage_model(input_tensor)

        def loss_func(output_tensor):
            loss = (output_tensor.float() - target.float()).square().mean()
            # Core divides the returned loss in place; logged values must not share that storage.
            return loss, {
                "microbatch": microbatch,
                "loss": loss.detach().clone(),
                "output": output_tensor.detach().clone(),
            }

        return output, loss_func

    # Use runtime's real communicator and group construction. No boundary double is installed for this case.
    losses = runtime.forward_backward_zero_bubble(
        forward_step_func=forward_step,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=2,
        seq_length=2,
        micro_batch_size=1,
    )

    with _device_runtime_rank_checks(case.device):
        assert model.weight.dtype == torch.bfloat16
        assert model.weight.main_grad.dtype == torch.float32
        actual_gradient = _assert_device_runtime_close(model.weight.main_grad, expected_gradient)
        for contribution in reference.contributions:
            assert not torch.allclose(actual_gradient, contribution[case.stage].float(), rtol=2e-2, atol=2e-4)
        expected_inputs = reference.inputs if case.stage == 0 else reference.hidden_states
        expected_outputs = reference.hidden_states if case.stage == 0 else reference.outputs
        assert len(model.forward_inputs) == len(model.forward_outputs) == 2
        for microbatch in range(2):
            _assert_device_runtime_close(model.forward_inputs[microbatch], expected_inputs[microbatch])
            _assert_device_runtime_close(model.forward_outputs[microbatch], expected_outputs[microbatch])
        if case.stage == 1:
            assert len(losses) == 2
            for microbatch, values in enumerate(losses):
                assert values["microbatch"] == microbatch
                _assert_device_runtime_close(values["loss"], reference.losses[microbatch])
                _assert_device_runtime_close(values["output"], reference.outputs[microbatch])
        else:
            assert losses == []
        assert case.store.should_split_bw is False
        assert case.store.cache == []
        assert case.store.weight_grad_queue is not None
        case.store.assert_empty()
