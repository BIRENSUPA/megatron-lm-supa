HAVE_TORCH_SUPA = False
try:
    import torch_supa  # noqa: F401

    HAVE_TORCH_SUPA = True
except ImportError:
    HAVE_TORCH_SUPA = False

if HAVE_TORCH_SUPA:
    from .megatron_mock import patch_br_torch

    patch_br_torch()

from .megatron_patch import exec_adaptation

exec_adaptation()
