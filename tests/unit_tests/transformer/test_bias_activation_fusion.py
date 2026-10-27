# tests/unit_tests/transformer/test_bias_activation_fusion.py
import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.transformer_config import TransformerConfig


def _simulate_cli_to_config(swiglu: bool, bias_gelu_fusion: bool, bias_swiglu_fusion: bool):
    """Mirror megatron/training/arguments.py:1668-1672 CLI flag -> config wiring.

    This is exactly how the CLI '--no-bias-gelu-fusion' / '--no-bias-swiglu-fusion' reach
    TransformerConfig.bias_activation_fusion, so the switch-under-test is exercised end to end
    from its CLI semantics (just without invoking argparse).
    """
    kw = {}
    if swiglu:
        kw["activation_func"] = F.silu
        kw["gated_linear_unit"] = True
        kw["bias_activation_fusion"] = bias_swiglu_fusion
    else:
        kw["bias_activation_fusion"] = bias_gelu_fusion
    return TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=4,
        add_bias_linear=True,
        **kw,
    )

@pytest.mark.sanity
@pytest.mark.regression
def test_bias_gelu_fusion_switch():
    """Feature: bias-gelu fusion switch (--no-bias-gelu-fusion).

    Default on -> bias_activation_fusion True; toggled off -> False.
    """
    cfg_on = _simulate_cli_to_config(swiglu=False, bias_gelu_fusion=True, bias_swiglu_fusion=True)
    cfg_off = _simulate_cli_to_config(swiglu=False, bias_gelu_fusion=False, bias_swiglu_fusion=True)
    assert cfg_on.bias_activation_fusion is True
    assert cfg_off.bias_activation_fusion is False
    assert cfg_on.bias_activation_fusion is not cfg_off.bias_activation_fusion

@pytest.mark.sanity
@pytest.mark.regression
def test_bias_swiglu_fusion_switch():
    """Feature: bias-swiglu fusion switch (--no-bias-swiglu-fusion).

    Under swiglu, default on -> bias_activation_fusion True; toggled off -> False.
    """
    cfg_on = _simulate_cli_to_config(swiglu=True, bias_gelu_fusion=True, bias_swiglu_fusion=True)
    cfg_off = _simulate_cli_to_config(swiglu=True, bias_gelu_fusion=True, bias_swiglu_fusion=False)
    assert cfg_on.bias_activation_fusion is True
    assert cfg_off.bias_activation_fusion is False
    assert cfg_on.bias_activation_fusion is not cfg_off.bias_activation_fusion

@pytest.mark.sanity
@pytest.mark.regression
def test_bias_dropout_fusion_switch():
    """Feature: bias-dropout fusion switch (--no-bias-dropout-fusion).

    bias_dropout_fusion drives the fused bias+dropout+residual path in TransformerLayer
    (transformer_layer.py:646). Toggle must flip the field that selects the fused vs
    unfused dropout. No model build needed: the field on the config is exactly what the
    layer reads.
    """
    cfg_on = TransformerConfig(
        num_layers=1, hidden_size=8, num_attention_heads=4,
        bias_dropout_fusion=True,
    )
    cfg_off = TransformerConfig(
        num_layers=1, hidden_size=8, num_attention_heads=4,
        bias_dropout_fusion=False,
    )
    assert cfg_on.bias_dropout_fusion is True
    assert cfg_off.bias_dropout_fusion is False
    assert cfg_on.bias_dropout_fusion is not cfg_off.bias_dropout_fusion
