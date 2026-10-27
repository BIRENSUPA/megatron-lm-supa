# tests/unit_tests/test_mixed_precision_switches.py
import pytest
import torch

from megatron.core.transformer.transformer_config import TransformerConfig


def _base_cfg(**kw):
    return TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=4,
        **kw,
    )

@pytest.mark.sanity
@pytest.mark.regression
def test_bf16_mode_switch():
    """Feature: BF16 low-precision mode (--bf16, mutually exclusive with --fp16).

    Default off; enabling sets ModelParallelConfig.bf16 True. bf16 and fp16 cannot be
    both True (transformer_config.py:1034).
    """
    cfg_default = _base_cfg()
    assert cfg_default.bf16 is False, "bf16 must default to False"

    cfg_on = _base_cfg(bf16=True, params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16)
    assert cfg_on.bf16 is True, "switch ON must set bf16 True"

    # Mutual exclusion: bf16 + fp16 must raise.
    with pytest.raises(ValueError):
        _base_cfg(bf16=True, fp16=True)

@pytest.mark.sanity
@pytest.mark.regression
def test_fp16_mode_switch():
    """Feature: FP16 low-precision mode (--fp16, mutually exclusive with --bf16).

    Default off; enabling sets ModelParallelConfig.fp16 True. fp16 and bf16 cannot be
    both True.
    """
    cfg_default = _base_cfg()
    assert cfg_default.fp16 is False, "fp16 must default to False"

    cfg_on = _base_cfg(fp16=True, params_dtype=torch.float16, pipeline_dtype=torch.float16)
    assert cfg_on.fp16 is True, "switch ON must set fp16 True"

    with pytest.raises(ValueError):
        _base_cfg(fp16=True, bf16=True)

@pytest.mark.sanity
@pytest.mark.regression
@pytest.mark.parametrize("fp8_format", ["e4m3", "hybrid"])
def test_fp8_mode_switch(fp8_format):
    """Feature: FP8 low-precision mode (--fp8-format e4m3|hybrid, default off).

    Default None (off); each allowed value is accepted by TransformerConfig.fp8.
    """
    cfg_default = _base_cfg()
    assert cfg_default.fp8 is None, "fp8 must default to None (off)"

    cfg_on = _base_cfg(fp8=fp8_format)
    assert cfg_on.fp8 == fp8_format, f"switch ON must set fp8 to {fp8_format}"
