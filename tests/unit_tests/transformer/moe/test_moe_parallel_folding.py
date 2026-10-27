import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


@pytest.mark.sanity
@pytest.mark.regression
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.internal
def test_moe_parallel_folding():
    """Feature: MoE parallel folding (expert_tensor_parallel_size decoupled from TP).

    Two ranks, attention TP=2 while MoE ETP=1 / EP=1. The expert sub-groups are built
    independently from the attention TP group (expert rank generator at parallel_state
    initialize_model_parallel), so attention and MoE share the same world but use
    different parallel decompositions. Asserts the decoupled groups exist and that a
    MoE forward/backward under the folded decomposition runs without error.
    """
    import megatron.core.parallel_state as parallel_state

    # Build the MoE layer inline (local spec) rather than via a shared test container,
    # so this case does not depend on upstream / third-party test code.
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
    )
    try:
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        config = TransformerConfig(
            tensor_model_parallel_size=2,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_tensor_parallel_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="allgather",
            moe_aux_loss_coeff=0.1,
            num_layers=1,
            moe_router_dtype="fp32",
            moe_grouped_gemm=False,
            hidden_size=16,
            num_attention_heads=8,
            use_cpu_initialization=False,
            sequence_parallel=True,
            add_bias_linear=False,
            moe_permute_fusion=False,
        )

        # Folding: expert tensor-parallel group is decoupled from the attention TP group.
        attn_tp = parallel_state.get_tensor_model_parallel_group()
        expt_tp = parallel_state.get_expert_tensor_parallel_group()
        assert expt_tp is not attn_tp, "expert TP group must be decoupled from attention TP group"
        assert attn_tp.size() == 2, "attention TP group must span 2 ranks"
        assert expt_tp.size() == 1, "expert TP group must be folded to size 1"

        # expert data-parallel size folds accordingly: world_size // (ETP * EP * PP) = 2 // 1 = 2.
        assert parallel_state.get_expert_data_parallel_world_size() == 2

        # MoE layer built under the folded decomposition; run a forward/backward smoke pass.
        spec = get_gpt_layer_local_spec(num_experts=8, moe_grouped_gemm=False)
        moe_layer = MoELayer(config, get_submodules(spec.submodules.mlp)).cuda()
        moe_layer.set_layer_number(0)

        bs, seql = 32, 8
        hidden_states = torch.randn(
            (bs, seql, config.hidden_size), dtype=torch.float32, device="cuda"
        )
        loss = moe_layer(hidden_states)[0].sum()
        loss.backward()  # must not raise under the folded (decoupled) parallel groups
    finally:
        torch.distributed.barrier()
        torch.cuda.synchronize()
        Utils.destroy_model_parallel()
