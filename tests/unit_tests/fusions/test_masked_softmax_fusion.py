# tests/unit_tests/fusions/test_masked_softmax_fusion.py
from unittest import mock

import pytest
import torch

from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.transformer.enums import AttnMaskType
from tests.unit_tests.test_utilities import Utils

@pytest.mark.sanity
@pytest.mark.regression
@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_masked_softmax_fusion_switch_takes_effect():
    """Feature: masked-softmax fusion switch (masked_softmax_fusion, --no-masked-softmax-fusion).

    FusedScaleMaskSoftmax.is_kernel_available gates the fused CUDA kernel on the
    masked_softmax_fusion field as its very first condition (fused_softmax.py:254).
    Two ranks: with the switch on, a fused-compatible shape reports the kernel available;
    with the switch off, the same shape reports unavailable. The toggle must flip the branch.

    get_batch_per_block imports the scaled_masked_softmax_cuda extension which is not built
    in this environment, so it is stubbed to a fixed batch-per-block value.
    """
    Utils.initialize_model_parallel(1, 1)

    # fused kernel constraints: fp16/bf16 input, 16 < sk <= 4096, sq%4==0, sk%4==0, b*np%4==0
    b, np_heads, sq, sk = 2, 2, 64, 64

    def _build(fusion_on: bool) -> FusedScaleMaskSoftmax:
        return FusedScaleMaskSoftmax(
            input_in_fp16=True,
            input_in_bf16=False,
            attn_mask_type=AttnMaskType.padding,
            scaled_masked_softmax_fusion=fusion_on,
            mask_func=lambda x, m: x,
            softmax_in_fp32=True,
            scale=None,
        )

    fused_on = _build(True)
    fused_off = _build(False)

    # Stub get_batch_per_block so it does not import the unavailable cuda extension.
    with mock.patch.object(FusedScaleMaskSoftmax, "get_batch_per_block", return_value=4):
        avail_on = fused_on.is_kernel_available(mask=None, b=b, np=np_heads, sq=sq, sk=sk)
        avail_off = fused_off.is_kernel_available(mask=None, b=b, np=np_heads, sq=sq, sk=sk)

    assert avail_on is True, "fusion ON should select the fused kernel branch"
    assert avail_off is False, "fusion OFF should select the torch softmax branch"
    assert avail_on is not avail_off, "switch did not change the selected path"

    Utils.destroy_model_parallel()
