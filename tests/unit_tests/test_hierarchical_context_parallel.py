# tests/unit_tests/test_hierarchical_context_parallel.py
import pytest
import torch.distributed as dist

import megatron.core.parallel_state as ps
from tests.unit_tests.test_utilities import Utils

@pytest.mark.sanity
@pytest.mark.regression
@pytest.mark.internal
def test_hierarchical_context_parallel_groups():
    """Feature: hierarchical context parallel.

    Two ranks, CP2 with hierarchical_context_parallel_sizes=[1, 2]. Exercises the
    real create_hierarchical_groups path under initialize_model_parallel and asserts
    that two distinct hierarchical CP sub-groups are created (level 0 size 1, level 1
    size 2) and that they differ from the flat CP group.
    """
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=2,
        hierarchical_context_parallel_sizes=[1, 2],
    )

    hcp_groups = ps.get_hierarchical_context_parallel_groups()
    assert hcp_groups is not None, "hierarchical CP groups were not created"
    assert len(hcp_groups) == 2, "expected two hierarchical CP levels"

    flat_cp = ps.get_context_parallel_group()
    flat_cp_ranks = set(dist.get_process_group_ranks(flat_cp))
    assert len(flat_cp_ranks) == 2, "flat CP group must span 2 ranks"

    level_sizes = sorted(g.size() for g in hcp_groups)
    assert level_sizes == [1, 2], f"expected level sizes [1, 2], got {level_sizes}"

    # The size-2 level must coincide with the flat CP ranks; the size-1 level is a
    # strict subset (the degenerate/inner level under the [1, 2] split).
    size2_group = next(g for g in hcp_groups if g.size() == 2)
    assert set(dist.get_process_group_ranks(size2_group)) == flat_cp_ranks

    size1_group = next(g for g in hcp_groups if g.size() == 1)
    assert set(dist.get_process_group_ranks(size1_group)).issubset(flat_cp_ranks)

    Utils.destroy_model_parallel()
