import importlib.util

from ..patchs.patch_utils import PatchManager

HAVE_TORCH_SUPA = False
try:
    import torch_supa  # noqa: F401

    HAVE_TORCH_SUPA = True
except ImportError:
    HAVE_TORCH_SUPA = False


def profile_patch():
    from .profiler_patch import apply_patches

    apply_patches()


def patch_gated_delta_net_causal_conv1d():
    # Use FLA only when the standalone extension is unavailable; overriding
    # the standalone path could change deterministic/concurrent-stream behavior.
    try:
        import causal_conv1d  # noqa: F401
        return
    except ImportError:
        pass
    try:
        from fla.modules.conv.causal_conv1d import causal_conv1d as _fla_causal_conv1d
    except ImportError:
        return

    def causal_conv1d_fn(x, weight, bias=None, activation=None, **_kwargs):
        # FLA expects [B, T, D]; standalone C ext expects [B, D, T]. The GDN
        # forward call site passes x as [B, D, T], so transpose to [B, T, D]
        # (contiguous) for fla, then transpose the output back to [B, D, T].
        out, _final_state = _fla_causal_conv1d(
            x=x.transpose(1, 2).contiguous(),
            weight=weight,
            bias=bias,
            activation=activation,
            output_final_state=False,
        )
        return out.transpose(1, 2)

    PatchManager.register_patch(
        'megatron.core.ssm.gated_delta_net.causal_conv1d_fn',
        causal_conv1d_fn,
    )


def exec_adaptation():
    profile_patch()
    patch_model_type()
    patch_megatron_core()
    patch_megatron_noncore()
    patch_gated_delta_net_causal_conv1d()
    PatchManager.apply_patches()


def patch_model_type():
    from .model_type_patch import patch_model_type

    patch_model_type()


def patch_megatron_core():
    if HAVE_TORCH_SUPA:
        patch_fused_a2a()
        patch_nccl_allocator()


def patch_megatron_noncore():
    if importlib.util.find_spec('megatron.training') is None:
        return
    patch_miscellaneous()
    patch_zero_bubble()


def patch_fused_a2a():
    from ..core.transformer.moe.fused_a2a import get_buffer

    PatchManager.register_patch('megatron.core.transformer.moe.fused_a2a.get_buffer', get_buffer)


def patch_nccl_allocator():
    from ..core.nccl_allocator import _build_nccl_allocator

    PatchManager.register_patch('megatron.core.nccl_allocator._build_nccl_allocator', _build_nccl_allocator)


def patch_zero_bubble():
    import sys

    import megatron.core as megatron_core
    import megatron.core.pipeline_parallel as pipeline_parallel
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.training.arguments import validate_args

    from ..core import zbpp_utils
    from ..core.pipeline_parallel import zerobubble
    from ..core.pipeline_parallel.schedules import get_forward_backward_func_decorator
    from ..core.pipeline_parallel.zerobubble import runtime as zerobubble_runtime
    from ..core.pipeline_parallel.zerobubble import scheduler as zerobubble_scheduler
    from ..core.pipeline_parallel.zerobubble.scheduler import communication as zerobubble_communication
    from ..core.pipeline_parallel.zerobubble.scheduler import graph as zerobubble_graph
    from ..core.pipeline_parallel.zerobubble.scheduler import passes as zerobubble_passes
    from ..core.pipeline_parallel.zerobubble.scheduler import zb as zerobubble_zb
    from ..core.tensor_parallel.layers import LinearWithGradAccumulationAndAsyncCommunication
    from ..training.arguments import validate_args_decorator

    sys.modules['megatron.core.zbpp_utils'] = zbpp_utils
    setattr(megatron_core, 'zbpp_utils', zbpp_utils)
    sys.modules['megatron.core.pipeline_parallel.zerobubble'] = zerobubble
    setattr(pipeline_parallel, 'zerobubble', zerobubble)
    sys.modules['megatron.core.pipeline_parallel.zerobubble.runtime'] = zerobubble_runtime
    sys.modules['megatron.core.pipeline_parallel.zerobubble.scheduler'] = zerobubble_scheduler
    sys.modules['megatron.core.pipeline_parallel.zerobubble.scheduler.communication'] = zerobubble_communication
    sys.modules['megatron.core.pipeline_parallel.zerobubble.scheduler.graph'] = zerobubble_graph
    sys.modules['megatron.core.pipeline_parallel.zerobubble.scheduler.passes'] = zerobubble_passes
    sys.modules['megatron.core.pipeline_parallel.zerobubble.scheduler.zb'] = zerobubble_zb

    PatchManager.register_patch(
        'megatron.core.pipeline_parallel.schedules.get_forward_backward_func',
        get_forward_backward_func_decorator(get_forward_backward_func),
    )
    PatchManager.register_patch(
        'megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication',
        LinearWithGradAccumulationAndAsyncCommunication,
    )
    PatchManager.register_patch('megatron.training.arguments.validate_args', validate_args_decorator(validate_args))


def patch_miscellaneous():
    from megatron.training.arguments import parse_args

    from ..training.arguments import parse_args_decorator

    PatchManager.register_patch('megatron.training.arguments.parse_args', parse_args_decorator(parse_args))
