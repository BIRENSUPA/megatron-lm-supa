# tests/unit_tests/test_activation_switches.py
from types import SimpleNamespace

import pytest

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig

@pytest.mark.sanity
@pytest.mark.regression
def test_recompute_activations_switch():
    """Feature: activation recompute switch (--recompute-activations, default off).

    Mirrors arguments.py:498-500: --recompute-activations maps to
    recompute_granularity='selective' (the flag itself is then deleted). The switch takes
    effect by flipping recompute_granularity from None to 'selective'.
    """
    # Off (default): granularity is None.
    cfg_off = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=4)
    assert cfg_off.recompute_granularity is None, "recompute must default to off (None)"

    # Simulate the CLI mapping exactly as arguments.py does it.
    args = SimpleNamespace(recompute_activations=True, recompute_granularity=None)
    if args.recompute_activations:
        args.recompute_granularity = "selective"
        del args.recompute_activations

    cfg_on = TransformerConfig(
        num_layers=1, hidden_size=8, num_attention_heads=4,
        recompute_granularity=args.recompute_granularity,
    )
    assert cfg_on.recompute_granularity == "selective", "switch ON must set selective recompute"
    assert not hasattr(args, "recompute_activations"), "flag is consumed into granularity"

@pytest.mark.sanity
@pytest.mark.regression
def test_cpu_offloading_num_layers_switch():
    """Feature: async activation offload switch (--cpu-offloading-num-layers N, N>0 triggers).

    Mirrors arguments.py:1555-1556: when cpu_offloading_num_layers > 0, cpu_offloading is
    forced True. ModelParallelConfig.cpu_offloading_num_layers is the field.
    """
    # N = 0 (default): offload not triggered.
    cfg_off = ModelParallelConfig(cpu_offloading_num_layers=0)
    assert cfg_off.cpu_offloading_num_layers == 0
    assert cfg_off.cpu_offloading is False, "N=0 must not trigger cpu offloading"

    # Simulate the CLI trigger rule exactly as arguments.py does it.
    args = SimpleNamespace(cpu_offloading_num_layers=4, cpu_offloading=False)
    if args.cpu_offloading_num_layers > 0:
        args.cpu_offloading = True

    cfg_on = ModelParallelConfig(
        cpu_offloading_num_layers=args.cpu_offloading_num_layers,
        cpu_offloading=args.cpu_offloading,
    )
    assert cfg_on.cpu_offloading is True, "N>0 must trigger cpu offloading"
    assert cfg_on.cpu_offloading_num_layers == 4
