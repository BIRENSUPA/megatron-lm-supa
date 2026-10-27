# tests/unit_tests/test_optimizer_switches.py
import pytest

from megatron.core.distributed.distributed_data_parallel_config import (
    DistributedDataParallelConfig,
)
from megatron.core.optimizer.optimizer_config import OptimizerConfig

@pytest.mark.sanity
@pytest.mark.regression
def test_use_distributed_optimizer_switch():
    """Feature: distributed optimizer switch (--use-distributed-optimizer, default off).

    Toggles DistributedDataParallelConfig.use_distributed_optimizer.
    """
    cfg_off = DistributedDataParallelConfig()
    assert cfg_off.use_distributed_optimizer is False, "must default to False"

    cfg_on = DistributedDataParallelConfig(use_distributed_optimizer=True)
    assert cfg_on.use_distributed_optimizer is True, "switch ON must set it True"
    assert cfg_on.use_distributed_optimizer is not cfg_off.use_distributed_optimizer

@pytest.mark.sanity
@pytest.mark.regression
def test_optimizer_cpu_offload_dependency_chain():
    """Feature: optimizer CPU/GPU hybrid switch (--optimizer-cpu-offload).

    Dependency chain (arguments.py:1488-1496 + optimizer_config.py:394-400):
      --optimizer-cpu-offload requires --use-precision-aware-optimizer, which requires
      --use-distributed-optimizer. The full chain must be accepted; breaking the chain
      (precision-aware without distributed optimizer) must raise.
    """
    # Full chain accepted.
    cfg_full = OptimizerConfig(
        optimizer="adam",
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=True,
        optimizer_cpu_offload=True,
    )
    assert cfg_full.optimizer_cpu_offload is True
    assert cfg_full.use_precision_aware_optimizer is True
    assert cfg_full.use_distributed_optimizer is True

    # Broken chain: precision-aware optimizer without distributed optimizer must raise.
    with pytest.raises(AssertionError):
        OptimizerConfig(
            optimizer="adam",
            use_distributed_optimizer=False,
            use_precision_aware_optimizer=True,
        )
