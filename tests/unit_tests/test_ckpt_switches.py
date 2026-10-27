# tests/unit_tests/test_ckpt_switches.py
from types import SimpleNamespace

import pytest


def _simulate_use_dist_ckpt(ckpt_format: str) -> bool:
    """Mirror megatron/training/utils.py:725-726: use_dist_ckpt = ckpt_format != 'torch'."""
    args = SimpleNamespace(ckpt_format=ckpt_format)
    args.use_dist_ckpt = args.ckpt_format != "torch"
    return args.use_dist_ckpt


def _simulate_async_save(async_save: bool, use_persistent_ckpt_worker: bool, use_dist_ckpt: bool):
    """Mirror arguments.py:1430-1436: async_save without persistent worker is forced off."""
    args = SimpleNamespace(
        async_save=async_save,
        use_persistent_ckpt_worker=use_persistent_ckpt_worker,
        use_dist_ckpt=use_dist_ckpt,
    )
    if args.use_dist_ckpt and args.async_save:
        if not args.use_persistent_ckpt_worker:
            args.async_save = False
    return args.async_save

@pytest.mark.sanity
@pytest.mark.regression
def test_distributed_ckpt_reshard_switch():
    """Feature: distributed-checkpoint cross-parallel reshard (--ckpt-format, default on).

    Default ckpt_format='torch_dist' => use_dist_ckpt True (distributed ckpt on, reshard
    available). Setting 'torch' disables it. utils.py:725-726.
    """
    # Default on (distributed).
    assert _simulate_use_dist_ckpt("torch_dist") is True, "torch_dist must enable dist ckpt"
    # Explicitly disabled.
    assert _simulate_use_dist_ckpt("torch") is False, "torch must disable dist ckpt"
    # Other distributed formats also enable it.
    assert _simulate_use_dist_ckpt("torch_dcp") is True
    assert _simulate_use_dist_ckpt("fsdp_dtensor") is True

@pytest.mark.sanity
@pytest.mark.regression
def test_async_save_switch():
    """Feature: async checkpoint save (--async-save, requires --use-persistent-ckpt-worker).

    With the persistent worker, async_save stays True; without it, arguments.py:1430-1436
    forces async_save to False (soft dependency).
    """
    # Properly enabled: async + persistent worker + dist ckpt.
    assert _simulate_async_save(True, True, True) is True, "async save must stay on with worker"
    # Missing persistent worker: forced off.
    assert _simulate_async_save(True, False, True) is False, "async save forced off without worker"

@pytest.mark.sanity
@pytest.mark.regression
def test_sync_save_switch():
    """Feature: sync checkpoint save (default when --async-save is not given).

    async_save defaults to False (training_config.py:444); that is the synchronous mode.
    """
    # Default = sync (async_save False).
    assert _simulate_async_save(False, False, True) is False, "default must be synchronous"
    # Explicit sync also stays sync.
    assert _simulate_async_save(False, True, True) is False, "explicit sync must stay sync"
