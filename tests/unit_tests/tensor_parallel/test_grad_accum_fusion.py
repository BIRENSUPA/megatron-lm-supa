# tests/unit_tests/tensor_parallel/test_grad_accum_fusion.py
from unittest import mock

import pytest
import torch

from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import init_method_normal
from tests.unit_tests.test_utilities import Utils

from ..env_checks import has_transformer_engine


def _cfg(grad_fusion: bool) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=4,
        tensor_model_parallel_size=1,
        gradient_accumulation_fusion=grad_fusion,
        sequence_parallel=False,
        bf16=False,
        params_dtype=torch.float,
        pipeline_dtype=torch.float,
    )

@pytest.mark.sanity
@pytest.mark.regression
@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not has_transformer_engine(), reason="Transformer Engine not available")
def test_gradient_accumulation_fusion_switch_takes_effect():
    """Feature: gradient-accumulation fusion switch (gradient_accumulation_fusion,
    toggled by --no-gradient-accumulation-fusion).

    The flag is forwarded to TransformerEngine as fuse_wgrad_accumulation
    (extensions/transformer_engine.py:823). Two ranks: toggling the switch must flip the
    fuse_wgrad_accumulation kwarg the TE Linear actually receives.
    """
    Utils.initialize_model_parallel(1, 1)
    init_fn = init_method_normal(0.02)
    te_linear = __import__("transformer_engine.pytorch", fromlist=["Linear"]).Linear
    real_te_init = te_linear.__init__
    captured = {}

    def fake_init(self, *a, **k):
        captured.update(k)
        return real_te_init(self, *a, **k)

    # Switch ON.
    with mock.patch("megatron.core.extensions.transformer_engine.te.pytorch.Linear.__init__", fake_init):
        TEColumnParallelLinear(
            8, 8, config=_cfg(True), init_method=init_fn,
            gather_output=False, bias=False, skip_bias_add=False, is_expert=False,
        )
    assert captured.get("fuse_wgrad_accumulation") is True, "switch ON must enable fuse_wgrad_accumulation"

    captured.clear()
    # Switch OFF.
    with mock.patch("megatron.core.extensions.transformer_engine.te.pytorch.Linear.__init__", fake_init):
        TEColumnParallelLinear(
            8, 8, config=_cfg(False), init_method=init_fn,
            gather_output=False, bias=False, skip_bias_add=False, is_expert=False,
        )
    assert captured.get("fuse_wgrad_accumulation") is False, "switch OFF must disable fuse_wgrad_accumulation"

    Utils.destroy_model_parallel()
