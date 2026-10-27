import typing

import torch

# Megatron Core v0.19 uses typing.override, which was added in Python 3.12.
if not hasattr(typing, 'override'):
    from typing_extensions import override

    typing.override = override

from . import patchs
