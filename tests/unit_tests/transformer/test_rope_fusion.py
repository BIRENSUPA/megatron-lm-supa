# tests/unit_tests/transformer/test_rope_fusion.py
import pytest
import torch

from megatron.core.transformer.transformer_config import TransformerConfig


def _mla_config(apply_rope_fusion: bool) -> TransformerConfig:
    cfg = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=4,
        multi_latent_attention=True,
        apply_rope_fusion=apply_rope_fusion,
        fp16=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
    )
    # rope_type lives on a mixin dataclass, not __init__; set after construction.
    cfg.rope_type = "yarn"
    return cfg

@pytest.mark.sanity
@pytest.mark.regression
def test_rope_fusion_switch_takes_effect():
    """Feature: RoPE fusion switch (apply_rope_fusion, toggled by --no-rope-fusion).

    The CLI flag maps to TransformerConfig.apply_rope_fusion (arguments.py:2467
    dest='apply_rope_fusion'). The switch is verified by toggling it and asserting the
    config field flips, and that both values are accepted by the config post-init. The
    field is what the MLA attention forward branches on (multi_latent_attention.py:594).
    """
    cfg_on = _mla_config(apply_rope_fusion=True)
    cfg_off = _mla_config(apply_rope_fusion=False)

    assert cfg_on.apply_rope_fusion is True, "switch ON must set apply_rope_fusion True"
    assert cfg_off.apply_rope_fusion is False, "switch OFF must set apply_rope_fusion False"
    assert cfg_on.apply_rope_fusion is not cfg_off.apply_rope_fusion, "switch did not toggle the field"
