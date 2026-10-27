import functools
from functools import partial

import torch
from megatron.core.parallel_state import get_global_memory_buffer
from megatron.core.tensor_parallel.layers import (
    dist_all_gather_func,
    dist_reduce_scatter_func,
)
from megatron.core.utils import is_torch_min_version, prepare_input_tensors_for_wgrad_compute

_grad_accum_fusion_available = True
try:
    import fused_weight_gradient_mlp_cuda
except ImportError:
    _grad_accum_fusion_available = False

try:
    import transformer_engine  # noqa
    from transformer_engine.pytorch.module.base import get_dummy_wgrad

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    if is_torch_min_version("2.4.0a0"):
        custom_fwd = partial(torch.amp.custom_fwd, device_type="cuda")
        custom_bwd = partial(torch.amp.custom_bwd, device_type="cuda")
    else:
        custom_fwd = torch.cuda.amp.custom_fwd
        custom_bwd = torch.cuda.amp.custom_bwd
except Exception:
    custom_fwd = torch.cuda.amp.custom_fwd
    custom_bwd = torch.cuda.amp.custom_bwd


class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """See linear_with_grad_accumulation_and_async_allreduce"""

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
        tp_group,
        gtp_remat_size=1,
    ):
        """Forward."""
        if gradient_accumulation_fusion and hasattr(weight, "main_grad"):
            main_grad = weight.main_grad
        else:
            main_grad = None
        ctx.save_for_backward(input, weight)
        ctx.main_grad = main_grad
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.sequence_parallel = sequence_parallel
        ctx.wgrad_deferral_limit = wgrad_deferral_limit
        ctx.grad_output_buffer = grad_output_buffer
        ctx.tp_group = tp_group
        ctx.gtp_remat_size = gtp_remat_size

        if ctx.gtp_remat_size > 1:
            weight = weight.all_gather_and_prefetch(fwd=True)

        if sequence_parallel:
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * tp_group.size()

            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            dist_all_gather_func(all_gather_buffer, input, group=tp_group)
            total_input = all_gather_buffer
        else:
            total_input = input

        output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        """Backward."""
        input, weight = ctx.saved_tensors
        main_grad = ctx.main_grad
        use_bias = ctx.use_bias
        sharded_weight = weight
        if ctx.gtp_remat_size > 1:
            weight = sharded_weight.all_gather_and_prefetch_bwd()
            ctx.gradient_accumulation_fusion = False
        grad_output_buffer = ctx.grad_output_buffer
        wgrad_deferral_limit = ctx.wgrad_deferral_limit
        handle = None
        tp_group = ctx.tp_group

        if ctx.gradient_accumulation_fusion:
            weight.main_grad = main_grad

        def prepare_for_wgrad_compute(_grad_output, _total_input, _handle):
            if ctx.sequence_parallel and _handle is not None:
                _handle.wait()
            return prepare_input_tensors_for_wgrad_compute(_grad_output, _total_input)

        def preprocess_wgrad(_grad_output, _input, async_op):
            _handle = None
            if ctx.sequence_parallel:
                dim_size = list(_input.size())
                dim_size[0] = dim_size[0] * tp_group.size()
                _total_input = get_global_memory_buffer().get_tensor(dim_size, _input.dtype, "mpu")
                _handle = dist_all_gather_func(_total_input, _input, group=tp_group, async_op=async_op)
            else:
                _total_input = _input
            return _grad_output, _total_input, _handle

        def process_wgrad(_weight, _grad_output, _total_input, _handle, wgrad_gemm_accum_func):
            _grad_output, _total_input = prepare_for_wgrad_compute(_grad_output, _total_input, _handle)
            wgrad_gemm_accum_func(_total_input, _grad_output, _weight.main_grad)

        from megatron_supa.core.zbpp_utils import WeightGradStore

        wgrad_compute = not WeightGradStore.split_bw()
        if grad_output_buffer is not None and wgrad_compute:
            if wgrad_deferral_limit == 0 or len(grad_output_buffer) < wgrad_deferral_limit:
                grad_output_buffer.append(grad_output)
                wgrad_compute = False

        if wgrad_compute:
            grad_output, total_input, handle = preprocess_wgrad(grad_output, input, async_op=wgrad_compute)
        grad_input = grad_output.matmul(weight)

        if wgrad_compute:
            grad_output, total_input = prepare_for_wgrad_compute(grad_output, total_input, handle)

        if ctx.allreduce_dgrad:
            handle = torch.distributed.all_reduce(grad_input, group=tp_group, async_op=True)

        if ctx.sequence_parallel:
            assert not ctx.allreduce_dgrad
            dim_size = list(input.size())
            sub_grad_input = torch.empty(
                dim_size, dtype=input.dtype, device=torch.cuda.current_device(), requires_grad=False
            )
            handle = dist_reduce_scatter_func(sub_grad_input, grad_input, group=tp_group, async_op=True)

        if ctx.gradient_accumulation_fusion:
            if wgrad_compute:
                if hasattr(weight, "__fsdp_param__"):
                    weight.main_grad = weight.get_main_grad()
                    torch.matmul(grad_output.t(), total_input, out=weight.main_grad)
                else:
                    if weight.main_grad.dtype == torch.float32:
                        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(total_input, grad_output, weight.main_grad)
                    elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(total_input, grad_output, weight.main_grad)
                    else:
                        raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")
            elif hasattr(weight, "__fsdp_param__"):
                raise RuntimeError("Zero Bubble split wgrad does not support Megatron-FSDP params")
            elif weight.main_grad.dtype == torch.float32:
                WeightGradStore.put(
                    weight,
                    functools.partial(preprocess_wgrad, grad_output, input),
                    functools.partial(
                        process_wgrad,
                        weight,
                        wgrad_gemm_accum_func=fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32,
                    ),
                )
            elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                WeightGradStore.put(
                    weight,
                    functools.partial(preprocess_wgrad, grad_output, input),
                    functools.partial(
                        process_wgrad,
                        weight,
                        wgrad_gemm_accum_func=fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16,
                    ),
                )
            else:
                raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")

            if hasattr(weight, "grad_added_to_main_grad"):
                if getattr(weight, "zero_out_wgrad", False):
                    if HAVE_TE:
                        grad_weight = get_dummy_wgrad(list(weight.main_grad.shape), input.dtype, zero=True)
                    else:
                        grad_weight = torch.zeros(
                            weight.main_grad.shape,
                            dtype=input.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                else:
                    if HAVE_TE:
                        grad_weight = get_dummy_wgrad(list(weight.main_grad.shape), input.dtype)
                    else:
                        grad_weight = torch.empty(
                            weight.main_grad.shape,
                            dtype=input.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                weight.grad_added_to_main_grad = True
            else:
                grad_weight = None
        else:
            grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.gtp_remat_size > 1 and grad_weight is not None:
            grad_weight = sharded_weight.wgrad_reduce_scatter(grad_weight)

        if ctx.sequence_parallel:
            handle.wait()
            return (sub_grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None)

        if ctx.allreduce_dgrad:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None
