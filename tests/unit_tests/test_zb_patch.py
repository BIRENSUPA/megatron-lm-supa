"""Register ZB patches and repair aliases without applying global patches."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import megatron.core as megatron_core
import pytest
from megatron.core import pipeline_parallel
from megatron.core.pipeline_parallel import schedules as core_schedules
from megatron.training import arguments as core_arguments

from megatron_supa.core import zbpp_utils
from megatron_supa.core.pipeline_parallel import zerobubble
from megatron_supa.core.pipeline_parallel.zerobubble import runtime, scheduler
from megatron_supa.core.pipeline_parallel.zerobubble.scheduler import communication, graph, passes, zb
from megatron_supa.core.tensor_parallel import layers
from megatron_supa.patchs import megatron_patch

pytestmark = [pytest.mark.sanity, pytest.mark.regression]


@pytest.fixture
def registry(monkeypatch):
    isolated = {}
    monkeypatch.setattr(megatron_patch.PatchManager, "_patch_info_collection", isolated)
    return isolated


def test_zero_bubble_repairs_module_aliases_and_registers_three_real_targets(registry, monkeypatch):
    aliases = {
        "megatron.core.zbpp_utils": zbpp_utils,
        "megatron.core.pipeline_parallel.zerobubble": zerobubble,
        "megatron.core.pipeline_parallel.zerobubble.runtime": runtime,
        "megatron.core.pipeline_parallel.zerobubble.scheduler": scheduler,
        "megatron.core.pipeline_parallel.zerobubble.scheduler.communication": communication,
        "megatron.core.pipeline_parallel.zerobubble.scheduler.graph": graph,
        "megatron.core.pipeline_parallel.zerobubble.scheduler.passes": passes,
        "megatron.core.pipeline_parallel.zerobubble.scheduler.zb": zb,
    }
    for name in aliases:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setattr(megatron_core, "zbpp_utils", object(), raising=False)
    monkeypatch.setattr(pipeline_parallel, "zerobubble", object(), raising=False)

    def original_selector(*args, **kwargs):
        raise AssertionError("registration must not execute the selector")

    def original_validate(*args, **kwargs):
        raise AssertionError("registration must not validate arguments")

    monkeypatch.setattr(core_schedules, "get_forward_backward_func", original_selector)
    monkeypatch.setattr(core_arguments, "validate_args", original_validate)
    megatron_patch.patch_zero_bubble()

    assert all(sys.modules[name] is module for name, module in aliases.items())
    assert megatron_core.zbpp_utils is zbpp_utils
    assert pipeline_parallel.zerobubble is zerobubble
    selector_target = "megatron.core.pipeline_parallel.schedules.get_forward_backward_func"
    linear_target = "megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication"
    validator_target = "megatron.training.arguments.validate_args"
    assert set(registry) == {selector_target, linear_target, validator_target}
    assert registry[selector_target].patch_func.__wrapped__ is original_selector
    assert registry[linear_target].patch_func is layers.LinearWithGradAccumulationAndAsyncCommunication
    assert registry[validator_target].patch_func.__wrapped__ is original_validate
    assert all(patch.is_applied is False for patch in registry.values())
    assert core_schedules.get_forward_backward_func is original_selector
    assert core_arguments.validate_args is original_validate


@pytest.mark.parametrize("training_present", [False, True], ids=["missing-training", "training-present"])
def test_noncore_checks_training_and_calls_misc_before_zero_bubble(registry, monkeypatch, training_present):
    calls = []
    find_spec = Mock(return_value=object() if training_present else None)
    monkeypatch.setattr(megatron_patch, "importlib", SimpleNamespace(util=SimpleNamespace(find_spec=find_spec)))
    miscellaneous = Mock(side_effect=lambda: calls.append("misc"))
    zero_bubble = Mock(side_effect=lambda: calls.append("zb"))
    monkeypatch.setattr(megatron_patch, "patch_miscellaneous", miscellaneous)
    monkeypatch.setattr(megatron_patch, "patch_zero_bubble", zero_bubble)

    megatron_patch.patch_megatron_noncore()

    find_spec.assert_called_once_with("megatron.training")
    assert calls == (["misc", "zb"] if training_present else [])
    if training_present:
        miscellaneous.assert_called_once_with()
        zero_bubble.assert_called_once_with()
    else:
        miscellaneous.assert_not_called()
        zero_bubble.assert_not_called()
    assert registry == {}


def test_miscellaneous_registers_parse_wrapper_bound_to_original_parser(registry, monkeypatch):
    def original_parse(*args, **kwargs):
        raise AssertionError("registration must not parse arguments")

    monkeypatch.setattr(core_arguments, "parse_args", original_parse)
    megatron_patch.patch_miscellaneous()

    target = "megatron.training.arguments.parse_args"
    assert set(registry) == {target}
    assert registry[target].patch_func.__wrapped__ is original_parse
    assert registry[target].is_applied is False
    assert core_arguments.parse_args is original_parse
