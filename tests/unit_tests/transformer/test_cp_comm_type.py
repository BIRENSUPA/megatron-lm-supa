# tests/unit_tests/transformer/test_cp_comm_type.py
import pytest

from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils


def _make_config():
    return TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        tensor_model_parallel_size=1,
        context_parallel_size=2,
    )

@pytest.mark.sanity
@pytest.mark.regression
@pytest.mark.internal
@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_te_min_version("1.10.0"), reason="TE >= 1.10.0 required for cp_comm_type")
@pytest.mark.parametrize("cp_comm_type", ["p2p", "all_gather", "a2a"])
def test_cp_comm_type_configurable(cp_comm_type):
    """Feature: configurable context-parallel communication type (cp_comm_type).

    Two ranks, CP2 over a real cp group. For each allowed value the attention layer
    is constructed through the real TE CP-setup path (cp_group, cp_stream, cp_comm_type
    wired at transformer_engine.py:1445-1463) and must store the configured value.
    """
    import torch

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=2,
    )

    config = _make_config()
    attn = TEDotProductAttention(
        config=config,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        cp_comm_type=cp_comm_type,
    )
    assert attn.cp_comm_type == cp_comm_type, "cp_comm_type not stored on the attention layer"

    Utils.destroy_model_parallel()

@pytest.mark.sanity
@pytest.mark.regression
@pytest.mark.internal
@pytest.mark.skipif(not is_te_min_version("1.10.0"), reason="TE >= 1.10.0 required for cp_comm_type")
def test_cp_comm_type_per_layer_list():
    """Feature: configurable CP comm type, per-layer list form.

    A list cp_comm_type (one entry per layer) must be accepted by TransformerConfig
    post-init validation when context_parallel_size > 1; a length-mismatched list
    must be rejected. Single-rank, no comm.
    """
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=2,
    )

    # Valid: list length matches num_layers.
    cfg_ok = TransformerConfig(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        tensor_model_parallel_size=1,
        context_parallel_size=2,
        cp_comm_type=["a2a", "p2p"],
    )
    assert cfg_ok.cp_comm_type == ["a2a", "p2p"]

    # Invalid: list length != num_layers.
    with pytest.raises(AssertionError):
        TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=4,
            tensor_model_parallel_size=1,
            context_parallel_size=2,
            cp_comm_type=["a2a"],
        )

    Utils.destroy_model_parallel()
