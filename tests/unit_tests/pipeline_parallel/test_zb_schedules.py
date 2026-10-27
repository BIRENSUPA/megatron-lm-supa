"""Select ZB only when enabled while preserving the ordinary selector's arguments."""

from types import SimpleNamespace

import pytest
from megatron.core.pipeline_parallel.zerobubble import runtime

from megatron_supa.core.pipeline_parallel import schedules

pytestmark = [pytest.mark.sanity, pytest.mark.regression]


@pytest.mark.parametrize("args_state", ["uninitialized", "none", "missing-switch", "disabled"])
@pytest.mark.parametrize("explicit_arguments", [False, True])
def test_falls_back_to_regular_selector(monkeypatch, args_state, explicit_arguments):
    def get_args():
        if args_state == "uninitialized":
            raise AssertionError("global args are not initialized")
        return {
            "none": None,
            "missing-switch": SimpleNamespace(),
            "disabled": SimpleNamespace(enable_zero_bubble=False),
        }[args_state]

    monkeypatch.setattr(schedules, "get_args", get_args)
    regular_result = object()
    calls = []

    def regular_selector(**kwargs):
        calls.append(kwargs)
        return regular_result

    selector = schedules.get_forward_backward_func_decorator(regular_selector)
    groups = object()
    options = dict(pp_size=4, vp_size=2, schedule_pg_collection=groups) if explicit_arguments else {}

    result = selector(**options)

    assert result is regular_result
    expected = options if explicit_arguments else dict(pp_size=None, vp_size=None, schedule_pg_collection=None)
    assert calls == [expected]
    if explicit_arguments:
        assert calls[0]["schedule_pg_collection"] is groups


def test_enabled_switch_returns_actual_zb_runtime(monkeypatch):
    monkeypatch.setattr(schedules, "get_args", lambda: SimpleNamespace(enable_zero_bubble=True))

    def regular_selector(**kwargs):
        pytest.fail("the ordinary selector must not run when zero bubble is enabled")

    selector = schedules.get_forward_backward_func_decorator(regular_selector)

    assert selector(pp_size=2, vp_size=None, schedule_pg_collection=object()) is runtime.forward_backward_zero_bubble
