"""Behavior of profiler annotations with real CPU tensors and autograd.

Load the leaf module without triggering accelerator adaptation in package
``__init__``. Optional Megatron JIT and FLA entrypoints are small CPU stand-ins;
these tests exercise their instrumentation, not their compiler or GPU kernels.
"""

import gzip
import importlib.machinery
import importlib.util
import inspect
import json
import sys
from functools import wraps
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.sanity, pytest.mark.regression]


@pytest.fixture
def profiler_patch(monkeypatch):
    path = Path(__file__).resolve().parents[2] / 'megatron_supa/patchs/profiler_patch.py'
    if not path.is_file():
        # CI packages tests separately from the installed adaptor wheel.
        package = importlib.machinery.PathFinder.find_spec('megatron_supa')
        assert package is not None and package.submodule_search_locations
        path = Path(next(iter(package.submodule_search_locations))) / 'patchs/profiler_patch.py'
    # Retain the package prefix for coverage's installed-wheel source matching.
    spec = importlib.util.spec_from_file_location('megatron_supa.patchs._profiler_patch_behavior_ut', path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    # Other collected tests may already have enabled trace naming. Temporarily
    # expose the real exporter, and restore that process's exact prior state.
    constructor = torch.profiler.profile
    profile_type = inspect.unwrap(constructor)
    exporter = inspect.unwrap(profile_type.export_chrome_trace)
    monkeypatch.setattr(profile_type, 'export_chrome_trace', exporter)
    if constructor is not profile_type:
        monkeypatch.setattr(constructor, 'export_chrome_trace', exporter)
    return module


def _install_modules(monkeypatch, *names):
    """Replace complete optional module trees, restoring sys.modules on exit."""
    modules = {}
    for name in names:
        parts = name.split('.')
        for length in range(1, len(parts) + 1):
            full_name = '.'.join(parts[:length])
            if full_name in modules:
                continue
            module = ModuleType(full_name)
            module.__package__ = full_name
            module.__path__ = []
            modules[full_name] = module
            monkeypatch.setitem(sys.modules, full_name, module)
            if length > 1:
                setattr(modules['.'.join(parts[: length - 1])], parts[length - 1], module)
    return modules


@pytest.fixture
def parallel_ranks(monkeypatch):
    modules = _install_modules(monkeypatch, 'megatron.core.mpu')
    mpu = modules['megatron.core.mpu']
    mpu.get_data_parallel_rank = lambda: 2
    mpu.get_pipeline_model_parallel_rank = lambda: 3
    mpu.get_tensor_model_parallel_rank = lambda: 4
    mpu.get_context_parallel_rank = lambda: 5
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 7)
    return mpu


def _cpu_profile():
    return torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], acc_events=True)


def _annotation_names(profile):
    return [event.name for event in profile.events() if event.name.startswith('nonaten.')]


@pytest.mark.parametrize(
    ('requested', 'written'),
    [
        ('worker.pt.trace.json', 'rank7_dp2_pp3_tp4_cp5.pt.trace.json'),
        ('nested/worker.pt.trace.json', 'nested/rank7_dp2_pp3_tp4_cp5.pt.trace.json'),
        ('nested/worker.json.gz', 'nested/rank7_dp2_pp3_tp4_cp5.pt.trace.json.gz'),
        ('nested/worker.pt.trace.json.gz', 'nested/rank7_dp2_pp3_tp4_cp5.pt.trace.json.gz'),
        ('keep.json', 'keep.json'),
    ],
)
def test_trace_naming_exports_readable_trace_in_requested_directory(
    profiler_patch, parallel_ranks, monkeypatch, tmp_path, requested, written
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'nested').mkdir()
    profiler_patch.patch_torch_profiler_trace_naming()
    patched_exporter = torch.profiler.profile.export_chrome_trace
    profiler_patch.patch_torch_profiler_trace_naming()
    assert torch.profiler.profile.export_chrome_trace is patched_exporter

    with _cpu_profile() as profile:
        with torch.profiler.record_function('naming-probe'):
            torch.ones(2, device='cpu').add_(1)
    assert profile.export_chrome_trace(requested) is None

    exported = tmp_path / written
    opener = gzip.open if exported.suffix == '.gz' else open
    with opener(exported, 'rt') as stream:
        trace = json.load(stream)
    assert any(event.get('name') == 'naming-probe' for event in trace['traceEvents'])
    assert sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob('*') if path.is_file()) == [written]


def test_trace_naming_reaches_instances_returned_by_wrapped_profile_factory(
    profiler_patch, parallel_ranks, monkeypatch, tmp_path
):
    profile_type = inspect.unwrap(torch.profiler.profile)

    @wraps(profile_type)
    def factory(*args, **kwargs):
        return profile_type(*args, **kwargs)

    # SUPA's constructor wrapper exposes copied class methods as attributes.
    # Assigning one of those attributes must also update the real instance method.
    factory.export_chrome_trace = profile_type.export_chrome_trace
    monkeypatch.setattr(torch.profiler, 'profile', factory)
    profiler_patch.patch_torch_profiler_trace_naming()

    with _cpu_profile() as profile:
        with torch.profiler.record_function('factory-probe'):
            torch.ones(2, device='cpu').add_(1)
    profile.export_chrome_trace(str(tmp_path / 'worker.pt.trace.json'))

    exported = tmp_path / 'rank7_dp2_pp3_tp4_cp5.pt.trace.json'
    trace = json.loads(exported.read_text())
    assert any(event.get('name') == 'factory-probe' for event in trace['traceEvents'])
    assert list(tmp_path.iterdir()) == [exported]
    assert factory.export_chrome_trace is profile_type.export_chrome_trace


@pytest.mark.parametrize('failure', ['missing_mpu', 'rank_query'])
def test_trace_naming_failure_exports_original_path(
    profiler_patch, parallel_ranks, monkeypatch, tmp_path, failure, caplog
):
    if failure == 'missing_mpu':
        del sys.modules['megatron.core'].mpu
        monkeypatch.setitem(sys.modules, 'megatron.core.mpu', None)
    else:

        def unavailable_rank():
            raise RuntimeError('process group unavailable')

        monkeypatch.setattr(torch.distributed, 'get_rank', unavailable_rank)

    profiler_patch.patch_torch_profiler_trace_naming()
    requested = tmp_path / 'original.pt.trace.json'
    with _cpu_profile() as profile:
        torch.ones(1, device='cpu')
    with caplog.at_level('DEBUG', logger=profiler_patch.__name__):
        profile.export_chrome_trace(str(requested))

    assert json.loads(requested.read_text())['traceEvents']
    assert list(tmp_path.iterdir()) == [requested]
    assert 'Failed to customize torch profiler trace name' in caplog.text


@pytest.mark.filterwarnings('ignore:Sparse CSR tensor support is in beta state:UserWarning')
def test_nested_parameters_encode_noncontiguous_sparse_scalar_and_constants(profiler_patch):
    transposed = torch.arange(6, dtype=torch.float32, device='cpu').reshape(2, 3).t()
    # CSR really raises on stride(), unlike COO, which reports zero strides.
    sparse = torch.tensor([[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]], device='cpu').to_sparse_csr()
    scalar = torch.tensor(4, dtype=torch.int64, device='cpu')
    params = profiler_patch.capture_params(
        (object(), [transposed, ([], sparse), 7], scalar),
        {'empty': (), 'sizes': [2, 3], 'enabled': True, 'optional': None},
        start_idx=1,
    )
    assert params == {
        'arg_1': {
            'type': 'TensorList',
            'tensors': {
                '[0]': {'shape': [3, 2], 'dtype': 'torch.float32', 'strides': [1, 3]},
                '[1][0]': {'type': 'EmptyList', 'count': 0},
                '[1][1]': {'shape': [2, 3], 'dtype': 'torch.float32', 'strides': None},
                '[2]': {'type': 'int', 'value': '7'},
            },
        },
        'arg_2': {'shape': [], 'dtype': 'torch.int64', 'strides': None},
        'empty': {'type': 'EmptyList', 'count': 0},
        'sizes': {'type': 'list', 'value': '[2, 3]'},
        'enabled': {'type': 'bool', 'value': 'True'},
        'optional': {'type': 'NoneType', 'value': 'None'},
    }
    assert profiler_patch.get_tensor_info(sparse) == {
        'shape': [2, 3],
        'dtype': 'torch.float32',
        'strides': None,
    }
    assert profiler_patch.format_profiler_name('nonaten.kernel', params) == (
        'nonaten.kernel|arg_1=TensorList[4]:{[0]=[3, 2]_Float_strides=[1, 3],'
        '[1][0]=EmptyList,[1][1]=[2, 3]_Float_strides=[],[2]=int}'
        '|arg_2=[]_Long_strides=[]|empty=EmptyList|sizes=list:[2, 3]'
        '|enabled=bool:True|optional=NoneType:None'
    )


@pytest.mark.parametrize(
    ('dtype', 'encoded'),
    [
        (torch.bfloat16, 'BFloat16'),
        (torch.float16, 'Half'),
        (torch.float32, 'Float'),
        (torch.float64, 'Double'),
        (torch.int64, 'Long'),
        (torch.int32, 'Int'),
        (torch.bool, 'Bool'),
        (torch.uint8, 'uint8'),
    ],
)
def test_tensor_dtype_is_encoded_without_losing_precision(profiler_patch, dtype, encoded):
    tensor = torch.ones((2, 3), dtype=dtype, device='cpu')
    if dtype == torch.float64:
        # Independent PyTorch JIT scalar-type oracle: Double is the C10 name
        # for this real tensor, rather than a guess from its dtype spelling.
        assert torch._C.TensorType.create_from_tensor(tensor).scalarType() == 'Double'
    params = profiler_patch.capture_params((tensor,), {})
    assert profiler_patch.format_profiler_name('nonaten.kernel', params) == (
        f'nonaten.kernel|arg_0=[2, 3]_{encoded}_strides=[3, 1]'
    )


def test_generic_wrapper_records_parameters_and_preserves_result_and_exception(profiler_patch):
    def affine(x, bias=0):
        if bias < 0:
            raise ValueError('negative bias')
        return x * 2 + bias

    wrapped = profiler_patch.create_wrapper(affine, 'nonaten.affine')
    x = torch.tensor([1.0, 3.0], requires_grad=True, device='cpu')
    with _cpu_profile() as profile:
        result = wrapped(x, bias=4)
        result.sum().backward()
        with pytest.raises(ValueError, match='negative bias'):
            wrapped(x, bias=-1)
    torch.testing.assert_close(result, torch.tensor([6.0, 10.0], device='cpu'))
    torch.testing.assert_close(x.grad, torch.tensor([2.0, 2.0], device='cpu'))
    assert _annotation_names(profile) == [
        'nonaten.affine|arg_0=[2]_Float_strides=[1]|bias=int:4',
        'nonaten.affine|arg_0=[2]_Float_strides=[1]|bias=int:-1',
    ]


@pytest.fixture
def jit_modules(profiler_patch, monkeypatch):
    modules = _install_modules(monkeypatch, 'megatron.core.jit', 'megatron.test_fusions', 'jit_consumer')

    def jit_fuser(func, factor=1, *, offset=0):
        # The stand-in makes compile options observable in outputs and grads,
        # so accidentally invoking the undecorated function cannot pass.
        @wraps(func)
        def compiled(*args, **kwargs):
            return func(*args, **kwargs) * factor + offset

        compiled._torchdynamo_orig_callable = func
        return compiled

    modules['megatron.core.jit'].jit_fuser = jit_fuser
    # Keep production discovery real, but confine its writes to disposable
    # modules; never traverse or replace functions in real torch/test modules.
    monkeypatch.setattr(profiler_patch, 'sys', SimpleNamespace(modules=modules))
    return modules


@pytest.mark.parametrize('form', ['direct', 'empty_factory', 'configured_factory', 'configured_direct'])
def test_jit_decorator_forms_execute_compiled_result_with_one_annotation(profiler_patch, jit_modules, form):
    jit_module = jit_modules['megatron.core.jit']
    assert profiler_patch.patch_jit_fuser() is True
    patched_decorator = jit_module.jit_fuser
    assert profiler_patch.patch_jit_fuser() is True
    assert jit_module.jit_fuser is patched_decorator

    def activation(x, bias=0):
        """Activation metadata survives decoration."""
        return x + bias

    activation.__module__ = 'megatron.test_fusions'
    if form == 'direct':
        wrapped = jit_module.jit_fuser(activation)
    elif form == 'empty_factory':
        wrapped = jit_module.jit_fuser()(activation)
    elif form == 'configured_factory':
        wrapped = jit_module.jit_fuser(factor=3, offset=1)(activation)
    else:
        wrapped = jit_module.jit_fuser(activation, 3, offset=1)

    x = torch.tensor([1.0, 2.0], requires_grad=True, device='cpu')
    with _cpu_profile() as profile:
        result = wrapped(x, bias=2)
        result.sum().backward()
    configured = form.startswith('configured')
    torch.testing.assert_close(result, torch.tensor([10.0, 13.0] if configured else [3.0, 4.0], device='cpu'))
    torch.testing.assert_close(x.grad, torch.tensor([3.0, 3.0] if configured else [1.0, 1.0], device='cpu'))
    assert wrapped.__name__ == 'activation'
    assert wrapped.__doc__ == 'Activation metadata survives decoration.'
    assert _annotation_names(profile) == [
        'nonaten.megatron.test_fusions.activation|arg_0=[2]_Float_strides=[1]|bias=int:2',
    ]


def test_jit_discovery_updates_predecorated_imports_without_double_wrapping(profiler_patch, jit_modules):
    jit_module = jit_modules['megatron.core.jit']
    fusions = jit_modules['megatron.test_fusions']
    consumer = jit_modules['jit_consumer']

    def activation(x):
        return x.square()

    activation.__module__ = fusions.__name__
    compiled = jit_module.jit_fuser(activation, factor=2)
    fusions.activation = compiled
    consumer.activation = compiled

    assert profiler_patch.patch_jit_fuser() is True
    profiler_patch.update_imported_references()
    assert consumer.activation is fusions.activation
    assert profiler_patch._patch_already_jit_fused_functions(jit_module.jit_fuser) == 0
    x = torch.tensor([2.0], requires_grad=True, device='cpu')
    with _cpu_profile() as profile:
        result = consumer.activation(x)
        result.sum().backward()
    torch.testing.assert_close(result, torch.tensor([8.0], device='cpu'))
    torch.testing.assert_close(x.grad, torch.tensor([8.0], device='cpu'))
    assert _annotation_names(profile) == [
        'nonaten.megatron.test_fusions.activation|arg_0=[1]_Float_strides=[1]',
    ]


def test_module_patching_handles_cycles_drifted_symbols_and_existing_imports(profiler_patch, monkeypatch):
    modules = _install_modules(
        monkeypatch,
        'transformer_engine.pytorch.cpp_extensions',
        'transformer_engine.kernels',
        'kernel_consumer',
    )
    extensions = modules['transformer_engine.pytorch.cpp_extensions']
    kernels = modules['transformer_engine.kernels']
    consumer = modules['kernel_consumer']

    def square(x):
        return x.square()

    square.__module__ = 'vendor_runtime'  # no longer discoverable by module ownership
    extensions.general_grouped_gemm = square
    extensions.grouped_gemm = len  # C/builtin entrypoints must stay untouched

    def double(x):
        return x * 2

    double.__module__ = kernels.__name__
    kernels.double = double
    extensions.kernels = kernels
    kernels.parent = extensions
    consumer.square = square
    consumer.double = double
    monkeypatch.setattr(profiler_patch, 'sys', SimpleNamespace(modules=modules))

    assert profiler_patch.patch_module(extensions.__name__) == 2
    profiler_patch.update_imported_references()
    assert profiler_patch.patch_module(extensions.__name__) == 0
    assert extensions.grouped_gemm is len
    with _cpu_profile() as profile:
        result = consumer.double(consumer.square(torch.tensor([3.0], device='cpu')))
    torch.testing.assert_close(result, torch.tensor([18.0], device='cpu'))
    assert _annotation_names(profile) == [
        'nonaten.transformer_engine.pytorch.cpp_extensions.gemm.general_grouped_gemm' '|arg_0=[1]_Float_strides=[1]',
        'nonaten.transformer_engine.kernels.double|arg_0=[1]_Float_strides=[1]',
    ]


@pytest.fixture
def fla_modules(monkeypatch):
    modules = _install_modules(
        monkeypatch,
        'fla.modules.l2norm',
        'fla.ops.gated_delta_rule',
        'fla.modules.conv.causal_conv1d',
    )

    def l2norm(x, eps=1e-6):
        return x.square()

    def chunk_gated_delta_rule(q, k, v, scale=1, output_final_state=False):
        output = ((q * k + v) * scale).sum(-1)
        return output, q.sum(0) if output_final_state else None

    def chunk_gated_delta_rule_refactored(x):
        return [None, x * 3, 'metadata', x.sum(-1)]

    def causal_conv1d(x, weight, bias=None, *, activation=None, output_final_state=False):
        return [x * weight + (0 if bias is None else bias), None]

    modules['fla.modules.l2norm'].l2norm = l2norm
    modules['fla.ops.gated_delta_rule'].chunk_gated_delta_rule = chunk_gated_delta_rule
    modules['fla.ops.gated_delta_rule'].chunk_gated_delta_rule_refactored = chunk_gated_delta_rule_refactored
    modules['fla.modules.conv.causal_conv1d'].causal_conv1d = causal_conv1d
    return modules


def test_fla_tensor_output_keeps_values_gradients_and_kernel_backward_label(profiler_patch, fla_modules):
    assert profiler_patch._patch_fla_autograd_functions() == 4
    assert profiler_patch._patch_fla_autograd_functions() == 0
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True, device='cpu')
    with _cpu_profile() as profile:
        result = fla_modules['fla.modules.l2norm'].l2norm(x)
        result.sum().backward()
    torch.testing.assert_close(result, torch.tensor([[1.0, 4.0], [9.0, 16.0]], device='cpu'))
    torch.testing.assert_close(x.grad, torch.tensor([[2.0, 4.0], [6.0, 8.0]], device='cpu'))
    assert _annotation_names(profile) == [
        'nonaten.fla.fla.modules.l2norm.l2norm|arg_0=[2, 2]_Float_strides=[2, 1]',
        'nonaten.fla.fla.modules.l2norm.l2norm_bwd|arg_0=[2, 2]_Float_strides=[2, 1]',
    ]


@pytest.mark.parametrize('final_state', [False, True])
def test_fla_tuple_keeps_optional_state_and_backward_uses_forward_input_shapes(
    profiler_patch, fla_modules, final_state
):
    profiler_patch._patch_fla_autograd_functions()
    q = torch.ones((2, 3), requires_grad=True, device='cpu')
    k = torch.full((2, 3), 2.0, requires_grad=True, device='cpu')
    v = torch.full((2, 3), 3.0, requires_grad=True, device='cpu')
    with _cpu_profile() as profile:
        result = fla_modules['fla.ops.gated_delta_rule'].chunk_gated_delta_rule(
            q,
            k,
            v,
            scale=2,
            output_final_state=final_state,
        )
        assert isinstance(result, tuple)
        loss = result[0].sum()
        if final_state:
            torch.testing.assert_close(result[1], torch.tensor([2.0, 2.0, 2.0], device='cpu'))
            loss = loss + result[1].sum()
        else:
            assert result[1] is None
        loss.backward()
    torch.testing.assert_close(result[0], torch.tensor([30.0, 30.0], device='cpu'))
    torch.testing.assert_close(q.grad, torch.full((2, 3), 5.0 if final_state else 4.0, device='cpu'))
    torch.testing.assert_close(k.grad, torch.full((2, 3), 2.0, device='cpu'))
    torch.testing.assert_close(v.grad, torch.full((2, 3), 2.0, device='cpu'))
    tensor_args = (
        '|arg_0=[2, 3]_Float_strides=[3, 1]' '|arg_1=[2, 3]_Float_strides=[3, 1]' '|arg_2=[2, 3]_Float_strides=[3, 1]'
    )
    assert _annotation_names(profile) == [
        'nonaten.fla.fla.ops.gated_delta_rule.chunk_gated_delta_rule'
        + tensor_args
        + f'|scale=int:2|output_final_state=bool:{final_state}',
        'nonaten.fla.fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule_bwd' + tensor_args,
    ]


def test_fla_list_keeps_nontensors_and_all_tensor_gradient_paths(profiler_patch, fla_modules):
    profiler_patch._patch_fla_autograd_functions()
    x = torch.tensor([[1.0, 2.0]], requires_grad=True, device='cpu')
    with _cpu_profile() as profile:
        result = fla_modules['fla.ops.gated_delta_rule'].chunk_gated_delta_rule_refactored(x)
        assert isinstance(result, list)
        assert result[0] is None
        assert result[2] == 'metadata'
        (result[1].sum() + result[3].sum()).backward()
    torch.testing.assert_close(result[1], torch.tensor([[3.0, 6.0]], device='cpu'))
    torch.testing.assert_close(result[3], torch.tensor([3.0], device='cpu'))
    torch.testing.assert_close(x.grad, torch.tensor([[4.0, 4.0]], device='cpu'))
    assert _annotation_names(profile) == [
        'nonaten.fla.fla.ops.gated_delta_rule.chunk_gated_delta_rule_refactored' '|arg_0=[1, 2]_Float_strides=[2, 1]',
        'nonaten.fla.fla.ops.gated_delta_rule.chunk_gated_delta_rule_refactored.backward'
        '|arg_0=[1, 2]_Float_strides=[2, 1]',
    ]


def test_fla_tensor_kwargs_follow_signature_order_and_keep_explicit_constants(profiler_patch, fla_modules):
    profiler_patch._patch_fla_autograd_functions()
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True, device='cpu')
    weight = torch.tensor([2.0, 3.0], requires_grad=True, device='cpu')
    with _cpu_profile() as profile:
        result = fla_modules['fla.modules.conv.causal_conv1d'].causal_conv1d(
            activation='identity',
            output_final_state=False,
            weight=weight,
            bias=None,
            x=x,
        )
        assert isinstance(result, list)
        assert result[1] is None
        result[0].sum().backward()
    torch.testing.assert_close(result[0], torch.tensor([[2.0, 6.0], [6.0, 12.0]], device='cpu'))
    torch.testing.assert_close(x.grad, torch.tensor([[2.0, 3.0], [2.0, 3.0]], device='cpu'))
    torch.testing.assert_close(weight.grad, torch.tensor([4.0, 6.0], device='cpu'))
    assert _annotation_names(profile) == [
        'nonaten.fla.fla.modules.conv.causal_conv1d.causal_conv1d'
        '|arg_0=[2, 2]_Float_strides=[2, 1]|arg_1=[2]_Float_strides=[1]'
        '|bias=NoneType:None|activation=str:identity|output_final_state=bool:False',
        'nonaten.fla.fla.modules.conv.triton.causal_conv1d_bwd'
        '|arg_0=[2, 2]_Float_strides=[2, 1]|arg_1=[2]_Float_strides=[1]',
    ]


def test_fla_uninspectable_signature_falls_back_to_actual_argument_order(profiler_patch, fla_modules):
    original = fla_modules['fla.modules.l2norm'].l2norm
    original.__signature__ = 'not inspectable'
    profiler_patch._patch_fla_autograd_functions()
    with _cpu_profile() as profile:
        result = fla_modules['fla.modules.l2norm'].l2norm(torch.tensor([2.0], device='cpu'), eps=0.25)
    torch.testing.assert_close(result, torch.tensor([4.0], device='cpu'))
    assert _annotation_names(profile) == [
        'nonaten.fla.fla.modules.l2norm.l2norm|arg_0=[1]_Float_strides=[1]|eps=float:0.25',
    ]


def test_fla_nontensor_output_is_returned_unchanged(profiler_patch, fla_modules):
    result = {'status': 'no tensor output'}
    fla_modules['fla.modules.l2norm'].l2norm = lambda: result
    profiler_patch._patch_fla_autograd_functions()
    with _cpu_profile() as profile:
        assert fla_modules['fla.modules.l2norm'].l2norm() is result
    assert _annotation_names(profile) == ['nonaten.fla.fla.modules.l2norm.l2norm']


def test_apply_patches_instruments_optional_modules_jit_fla_and_imported_references(
    profiler_patch, jit_modules, fla_modules, monkeypatch, tmp_path
):
    module_specs = (
        'transformer_engine_torch',
        'transformer_engine.pytorch.cpp_extensions',
        'apex.transformer.functional',
        'flash_attn',
        'fused_rotary_positional_embedding',
        'fused_weight_gradient_mlp_cuda',
    )
    modules = _install_modules(monkeypatch, *module_specs, 'instrumentation_consumer')
    consumer = modules['instrumentation_consumer']

    def add_constant(amount, module_name):
        def run(x):
            return x + amount

        run.__module__ = module_name
        return run

    for amount, module_name in enumerate(module_specs, start=1):
        function = add_constant(amount, module_name)
        modules[module_name].run = function
        setattr(consumer, f'operation_{amount}', function)

    # Keep JIT and rank accessors on the same disposable Megatron module tree.
    jit_modules['megatron.core'].mpu = SimpleNamespace(
        get_data_parallel_rank=lambda: 2,
        get_pipeline_model_parallel_rank=lambda: 3,
        get_tensor_model_parallel_rank=lambda: 4,
        get_context_parallel_rank=lambda: 5,
    )
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 7)
    consumer.jit_fuser = jit_modules['megatron.core.jit'].jit_fuser
    consumer.l2norm = fla_modules['fla.modules.l2norm'].l2norm
    monkeypatch.setattr(profiler_patch, 'sys', SimpleNamespace(modules={**jit_modules, **fla_modules, **modules}))

    profiler_patch.apply_patches()
    profiler_patch.apply_patches()

    def activation(x):
        return x + 1

    activation.__module__ = 'megatron.test_fusions'
    fused_activation = consumer.jit_fuser(activation)
    x = torch.tensor([1.0], requires_grad=True, device='cpu')
    with _cpu_profile() as profile:
        result = x
        for amount in range(1, 7):
            result = getattr(consumer, f'operation_{amount}')(result)
        result = fused_activation(consumer.l2norm(result))
        result.sum().backward()

    torch.testing.assert_close(result, torch.tensor([485.0], device='cpu'))
    torch.testing.assert_close(x.grad, torch.tensor([44.0], device='cpu'))
    assert _annotation_names(profile) == [
        'nonaten.transformer_engine_torch.run|arg_0=[1]_Float_strides=[1]',
        'nonaten.transformer_engine.pytorch.cpp_extensions.run|arg_0=[1]_Float_strides=[1]',
        'nonaten.apex.transformer.functional.run|arg_0=[1]_Float_strides=[1]',
        'nonaten.flash_attn.run|arg_0=[1]_Float_strides=[1]',
        'nonaten.fused_rotary_positional_embedding.run|arg_0=[1]_Float_strides=[1]',
        'nonaten.fused_weight_gradient_mlp_cuda.run|arg_0=[1]_Float_strides=[1]',
        'nonaten.fla.fla.modules.l2norm.l2norm|arg_0=[1]_Float_strides=[1]',
        'nonaten.megatron.test_fusions.activation|arg_0=[1]_Float_strides=[1]',
        'nonaten.fla.fla.modules.l2norm.l2norm_bwd|arg_0=[1]_Float_strides=[1]',
    ]
    profile.export_chrome_trace(str(tmp_path / 'worker.pt.trace.json'))
    exported = tmp_path / 'rank7_dp2_pp3_tp4_cp5.pt.trace.json'
    trace = json.loads(exported.read_text())
    assert any(
        event.get('name') == 'nonaten.flash_attn.run|arg_0=[1]_Float_strides=[1]' for event in trace['traceEvents']
    )


@pytest.mark.parametrize(
    ('unavailable', 'diagnostic'),
    [
        ('module', 'megatron.core.jit not available'),
        ('symbol', 'jit_fuser not found'),
        ('initialization', 'compiler backend unavailable'),
    ],
)
def test_jit_unavailable_dependency_is_reported_and_can_be_retried(
    profiler_patch, jit_modules, monkeypatch, caplog, unavailable, diagnostic
):
    core = jit_modules['megatron.core']
    jit_module = jit_modules['megatron.core.jit']
    base_jit_fuser = jit_module.jit_fuser
    if unavailable == 'module':
        monkeypatch.delattr(core, 'jit')
        monkeypatch.setitem(sys.modules, 'megatron.core.jit', None)
    elif unavailable == 'symbol':
        monkeypatch.delattr(jit_module, 'jit_fuser')
    else:

        class UnavailableCompiler(ModuleType):
            def __getattr__(self, name):
                if name == 'jit_fuser':
                    raise RuntimeError('compiler backend unavailable')
                raise AttributeError(name)

        broken = UnavailableCompiler('megatron.core.jit')
        monkeypatch.setattr(core, 'jit', broken)
        monkeypatch.setitem(sys.modules, 'megatron.core.jit', broken)

    with caplog.at_level('DEBUG', logger=profiler_patch.__name__):
        assert profiler_patch.patch_jit_fuser() is False
    assert diagnostic in caplog.text
    assert profiler_patch._JIT_FUSER_PATCHED is False

    # A failed optional dependency must not suppress a later successful patch.
    monkeypatch.setattr(core, 'jit', jit_module, raising=False)
    monkeypatch.setitem(sys.modules, 'megatron.core.jit', jit_module)
    monkeypatch.setattr(jit_module, 'jit_fuser', base_jit_fuser, raising=False)
    assert profiler_patch.patch_jit_fuser() is True

    def activation(x):
        return x.square()

    activation.__module__ = 'megatron.test_fusions'
    wrapped = jit_module.jit_fuser(activation)
    with _cpu_profile() as profile:
        result = wrapped(torch.tensor([3.0], device='cpu'))
    torch.testing.assert_close(result, torch.tensor([9.0], device='cpu'))
    assert _annotation_names(profile) == [
        'nonaten.megatron.test_fusions.activation|arg_0=[1]_Float_strides=[1]',
    ]


def test_missing_optional_module_returns_zero_and_can_later_be_instrumented(profiler_patch, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, 'flash_attn', None)
    with caplog.at_level('DEBUG', logger=profiler_patch.__name__):
        assert profiler_patch.patch_module('flash_attn') == 0
    assert 'flash_attn not available' in caplog.text

    modules = _install_modules(monkeypatch, 'flash_attn')

    def attention(x):
        return x * 3

    attention.__module__ = 'flash_attn'
    modules['flash_attn'].attention = attention
    assert profiler_patch.patch_module('flash_attn') == 1
    with _cpu_profile() as profile:
        result = modules['flash_attn'].attention(torch.tensor([2.0], device='cpu'))
    torch.testing.assert_close(result, torch.tensor([6.0], device='cpu'))
    assert _annotation_names(profile) == ['nonaten.flash_attn.attention|arg_0=[1]_Float_strides=[1]']


def test_jit_discovery_profiles_real_script_function_and_preserves_autograd(profiler_patch, jit_modules):
    compiled = torch.jit.CompilationUnit('def activation(x: Tensor) -> Tensor:\n    return x * x\n').activation
    jit_modules['megatron.test_fusions'].activation = compiled
    jit_modules['jit_consumer'].activation = compiled
    assert profiler_patch.patch_jit_fuser() is True
    profiler_patch.update_imported_references()
    assert profiler_patch._patch_already_jit_fused_functions(jit_modules['megatron.core.jit'].jit_fuser) == 0

    x = torch.tensor([2.0, 3.0], requires_grad=True, device='cpu')
    with _cpu_profile() as profile:
        result = jit_modules['jit_consumer'].activation(x)
        result.sum().backward()
    torch.testing.assert_close(result, torch.tensor([4.0, 9.0], device='cpu'))
    torch.testing.assert_close(x.grad, torch.tensor([4.0, 6.0], device='cpu'))
    names = _annotation_names(profile)
    assert len(names) == 1
    assert names[0].endswith('.activation|arg_0=[2]_Float_strides=[1]')


def test_jit_discovery_follows_wrapper_chain_without_dropping_outer_behavior(profiler_patch, jit_modules):
    def activation(x):
        return x.square()

    activation.__module__ = 'megatron.test_fusions'
    compiled = jit_modules['megatron.core.jit'].jit_fuser(activation, factor=2)

    @wraps(compiled, updated=())
    def middle(x):
        return compiled(x) + 5

    @wraps(middle, updated=())
    def outer(x):
        return middle(x) * 3

    @wraps(activation, updated=())
    def ordinary_wrapper(x):
        return activation(x)

    fusions = jit_modules['megatron.test_fusions']
    fusions.activation = outer
    fusions.ordinary = ordinary_wrapper
    assert profiler_patch.patch_jit_fuser() is True
    assert fusions.ordinary is ordinary_wrapper
    x = torch.tensor([2.0], requires_grad=True, device='cpu')
    with _cpu_profile() as profile:
        result = fusions.activation(x)
        result.sum().backward()
        ordinary_result = fusions.ordinary(x.detach())
    torch.testing.assert_close(result, torch.tensor([39.0], device='cpu'))
    torch.testing.assert_close(x.grad, torch.tensor([24.0], device='cpu'))
    torch.testing.assert_close(ordinary_result, torch.tensor([4.0], device='cpu'))
    assert _annotation_names(profile) == [
        'nonaten.megatron.test_fusions.activation|arg_0=[1]_Float_strides=[1]',
    ]


@pytest.mark.parametrize('explicit_scale', [False, True])
def test_fla_variadic_inputs_and_extra_kwargs_keep_order_and_omit_unpassed_defaults(
    profiler_patch, fla_modules, explicit_scale
):
    def variadic_l2norm(x, *others, scale=2, unused=None, **options):
        result = x
        for other in others:
            result = result + other
        return result * scale + options['offset']

    fla_modules['fla.modules.l2norm'].l2norm = variadic_l2norm
    profiler_patch._patch_fla_autograd_functions()
    x = torch.tensor([1.0, 2.0], requires_grad=True, device='cpu')
    y = torch.tensor([3.0, 4.0], requires_grad=True, device='cpu')
    z = torch.tensor([5.0, 6.0], requires_grad=True, device='cpu')
    offset = torch.tensor([7.0, 8.0], requires_grad=True, device='cpu')
    kwargs = {'offset': offset, 'label': 'diagnostic'}
    if explicit_scale:
        kwargs['scale'] = 3
    with _cpu_profile() as profile:
        result = fla_modules['fla.modules.l2norm'].l2norm(x, y, z, **kwargs)
        result.sum().backward()
    torch.testing.assert_close(result, torch.tensor([34.0, 44.0] if explicit_scale else [25.0, 32.0], device='cpu'))
    for tensor in (x, y, z):
        torch.testing.assert_close(tensor.grad, torch.full((2,), 3.0 if explicit_scale else 2.0, device='cpu'))
    torch.testing.assert_close(offset.grad, torch.tensor([1.0, 1.0], device='cpu'))
    leading_tensors = (
        '|arg_0=[2]_Float_strides=[1]' '|arg_1=TensorList[2]:{[0]=[2]_Float_strides=[1],[1]=[2]_Float_strides=[1]}'
    )
    offset_tensor = '|arg_2=[2]_Float_strides=[1]'
    scale = '|scale=int:3' if explicit_scale else ''
    assert _annotation_names(profile) == [
        'nonaten.fla.fla.modules.l2norm.l2norm' + leading_tensors + scale + offset_tensor + '|label=str:diagnostic',
        'nonaten.fla.fla.modules.l2norm.l2norm_bwd' + leading_tensors + offset_tensor,
    ]


def test_fla_unavailable_entrypoint_does_not_prevent_remaining_instrumentation(
    profiler_patch, fla_modules, monkeypatch, caplog
):
    class UnavailableEntrypoint(ModuleType):
        def __getattr__(self, name):
            if name == 'l2norm':
                raise RuntimeError('optional kernel unavailable')
            raise AttributeError(name)

    broken = UnavailableEntrypoint('fla.modules.l2norm')
    monkeypatch.setitem(sys.modules, 'fla.modules.l2norm', broken)
    fla_modules['fla.modules'].l2norm = broken
    with caplog.at_level('DEBUG', logger=profiler_patch.__name__):
        assert profiler_patch._patch_fla_autograd_functions() == 3
    assert 'optional kernel unavailable' in caplog.text

    x = torch.tensor([2.0], requires_grad=True, device='cpu')
    weight = torch.tensor([3.0], requires_grad=True, device='cpu')
    with _cpu_profile() as profile:
        result = fla_modules['fla.modules.conv.causal_conv1d'].causal_conv1d(x, weight)
        result[0].sum().backward()
    torch.testing.assert_close(result[0], torch.tensor([6.0], device='cpu'))
    torch.testing.assert_close(x.grad, torch.tensor([3.0], device='cpu'))
    torch.testing.assert_close(weight.grad, torch.tensor([2.0], device='cpu'))
    assert _annotation_names(profile) == [
        'nonaten.fla.fla.modules.conv.causal_conv1d.causal_conv1d'
        '|arg_0=[1]_Float_strides=[1]|arg_1=[1]_Float_strides=[1]',
        'nonaten.fla.fla.modules.conv.triton.causal_conv1d_bwd'
        '|arg_0=[1]_Float_strides=[1]|arg_1=[1]_Float_strides=[1]',
    ]
