"""Zero Bubble argument providers and validation wrappers."""

import argparse
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron_supa.training import arguments

pytestmark = [pytest.mark.sanity, pytest.mark.regression]


@pytest.fixture
def valid_args():
    return SimpleNamespace(
        enable_zero_bubble=True,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=None,
        overlap_grad_reduce=False,
        gradient_accumulation_fusion=True,
        transformer_impl="local",
        zero_bubble_max_pending_backward="auto",
    )


@pytest.mark.parametrize(
    "argv,enabled,pending",
    [([], False, "auto"), (["--enable-zero-bubble", "--zero-bubble-max-pending-backward", "7"], True, "7")],
)
def test_real_parser_defaults_and_explicit_zero_bubble_arguments(argv, enabled, pending):
    parser = argparse.ArgumentParser()
    assert arguments.process_br_args(parser) is parser
    args = parser.parse_args(argv)
    assert args.enable_zero_bubble is enabled
    assert args.zero_bubble_max_pending_backward == pending


@pytest.mark.parametrize("use_provider", [False, True], ids=["none", "replacement-parser"])
def test_extra_provider_keeps_the_parser_returned_by_the_original_provider(use_provider):
    original, replacement = argparse.ArgumentParser(), argparse.ArgumentParser()
    replacement.add_argument("--provided", default="from-provider")
    provider = Mock(return_value=replacement) if use_provider else None
    parser = arguments.extra_args_provider_decorator(provider)(original)
    assert parser is (replacement if use_provider else original)
    args = parser.parse_args(["--enable-zero-bubble"])
    assert args.enable_zero_bubble is True and args.zero_bubble_max_pending_backward == "auto"
    if use_provider:
        provider.assert_called_once_with(original)
        assert args.provided == "from-provider"


@pytest.mark.parametrize("ignore_unknown", [False, True])
def test_parse_wrapper_forwards_ignore_unknown_and_supplies_decorated_provider(ignore_unknown):
    calls = []

    def provider(parser):
        parser.add_argument("--provided", default=11, type=int)
        return parser

    def original_parse(extra_args_provider, ignore_unknown_args):
        calls.append((extra_args_provider, ignore_unknown_args))
        parser = extra_args_provider(argparse.ArgumentParser())
        return parser.parse_args(["--enable-zero-bubble", "--provided", "13"])

    decorated = arguments.parse_args_decorator(original_parse)
    parsed = decorated(extra_args_provider=provider, ignore_unknown_args=ignore_unknown)
    assert decorated.__wrapped__ is original_parse
    assert len(calls) == 1 and calls[0][1] is ignore_unknown
    assert calls[0][0] is not provider and calls[0][0].__wrapped__ is provider
    assert parsed.provided == 13 and parsed.enable_zero_bubble is True
    assert parsed.zero_bubble_max_pending_backward == "auto"


def test_validate_uses_original_validator_result_and_passes_defaults_by_identity(valid_args):
    original_args, defaults = object(), {"sentinel": object()}
    original_validate = Mock(return_value=valid_args)
    result = arguments.validate_args_decorator(original_validate)(original_args, defaults)
    original_validate.assert_called_once_with(original_args, defaults)
    assert original_validate.call_args.args[1] is defaults
    assert result is valid_args


def test_disabled_zero_bubble_does_not_apply_any_zb_restrictions(valid_args):
    valid_args.enable_zero_bubble = False
    valid_args.pipeline_model_parallel_size = 1
    valid_args.virtual_pipeline_model_parallel_size = 2
    valid_args.overlap_grad_reduce = True
    valid_args.gradient_accumulation_fusion = False
    valid_args.transformer_impl = "transformer_engine"
    valid_args.zero_bubble_max_pending_backward = "not-an-int"
    original_validate = Mock(return_value=valid_args)
    assert arguments.validate_args_decorator(original_validate)(valid_args) is valid_args
    original_validate.assert_called_once_with(valid_args, {})
    assert valid_args.zero_bubble_max_pending_backward == "not-an-int"


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("pipeline_model_parallel_size", 1, "requires pipeline-model-parallel-size > 1"),
        ("virtual_pipeline_model_parallel_size", 2, "does not support virtual pipeline parallelism"),
        ("overlap_grad_reduce", True, "does not support --overlap-grad-reduce"),
        ("gradient_accumulation_fusion", False, "requires --gradient-accumulation-fusion"),
        ("transformer_impl", "transformer_engine", "does not support transformer_engine"),
    ],
)
def test_enabled_zero_bubble_rejects_each_unsupported_configuration(valid_args, field, value, message):
    setattr(valid_args, field, value)
    original_validate = Mock(return_value=valid_args)
    with pytest.raises(AssertionError, match=message):
        arguments.validate_args_decorator(original_validate)(valid_args)
    original_validate.assert_called_once_with(valid_args, {})


@pytest.mark.parametrize("pending,expected", [("auto", "auto"), ("7", 7)])
def test_pending_backward_auto_is_preserved_and_numeric_text_is_converted(valid_args, pending, expected):
    valid_args.zero_bubble_max_pending_backward = pending
    result = arguments.validate_args_decorator(lambda args, defaults: args)(valid_args)
    assert result is valid_args
    assert result.zero_bubble_max_pending_backward == expected
    assert type(result.zero_bubble_max_pending_backward) is type(expected)


def test_pending_backward_rejects_non_numeric_text(valid_args):
    valid_args.zero_bubble_max_pending_backward = "not-an-int"
    with pytest.raises(ValueError, match="invalid literal for int"):
        arguments.validate_args_decorator(lambda args, defaults: args)(valid_args)
