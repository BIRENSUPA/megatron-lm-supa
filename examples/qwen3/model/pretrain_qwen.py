from functools import partial

import megatron_supa  # noqa: F401  # must be imported first to apply the SUPA patches

from megatron.core.enums import ModelType
from megatron.training import pretrain
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import parse_and_validate_args

# Reuse the upstream Megatron-LM entrypoint implementations instead of keeping a
# drifting local copy. These modules live at the root of third-party/megatron-lm,
# which the launcher adds to PYTHONPATH.
from gpt_builders import gpt_builder
from model_provider import model_provider
from pretrain_gpt import forward_step, train_valid_test_datasets_provider

if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True

    args = parse_and_validate_args()
    full_config = pretrain_cfg_container_from_args(args)
    pretrain(
        full_config,
        train_valid_test_datasets_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        model_provider=partial(model_provider, gpt_builder),
    )
