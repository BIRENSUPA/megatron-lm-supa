"""Exercise backend queries without initializing an accelerator communicator."""

import pytest
import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d

import megatron_supa  # noqa: F401
from megatron_supa.patchs import HAVE_TORCH_SUPA

pytestmark = [
    pytest.mark.sanity,
    pytest.mark.regression,
    pytest.mark.skipif(not HAVE_TORCH_SUPA, reason='BCCL compatibility requires torch_supa'),
]


@pytest.fixture
def registered_group(monkeypatch):
    # get_backend reads this registry; no collective or device allocation is needed.
    group = object()
    monkeypatch.setitem(c10d._world.pg_map, group, ('bccl', None))
    monkeypatch.setattr(c10d, '_get_default_group', lambda: group)
    return group


@pytest.mark.parametrize('query', [dist.get_backend, c10d.get_backend])
def test_bccl_queries_use_nccl_compatibility_name(registered_group, query):
    assert query() == 'nccl'
    assert query(registered_group) == 'nccl'
    assert query(group=registered_group) == 'nccl'
    assert c10d._world.pg_map[registered_group][0] == 'bccl'


def test_backend_imported_after_adaptation_uses_nccl_name(registered_group):
    from torch.distributed.distributed_c10d import get_backend

    assert get_backend(registered_group) == 'nccl'


@pytest.mark.parametrize('backend', ['gloo', 'nccl', 'fake', 'ucc'])
def test_other_backends_keep_their_names(registered_group, monkeypatch, backend):
    monkeypatch.setitem(c10d._world.pg_map, registered_group, (backend, None))
    assert dist.get_backend(registered_group) == backend


def test_invalid_groups_keep_original_errors():
    with pytest.raises(ValueError):
        dist.get_backend(dist.GroupMember.NON_GROUP_MEMBER)
    with pytest.raises(ValueError):
        dist.get_backend(object())


def test_uninitialized_default_group_keeps_original_error():
    assert not dist.is_initialized()
    with pytest.raises(ValueError):
        dist.get_backend()


def test_nvrx_timeouts_select_accelerator_for_bccl(registered_group):
    timeouts_calc = pytest.importorskip('nvidia_resiliency_ext.fault_tolerance.timeouts_calc')
    assert timeouts_calc.TimeoutsCalc()._device == torch.device('cuda')
