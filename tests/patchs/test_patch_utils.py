"""Exercise patch mechanics without executing package-wide accelerator adaptation."""

import importlib
import importlib.util
import sys
from importlib.machinery import PathFinder
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import pytest

pytestmark = [pytest.mark.sanity, pytest.mark.regression]

PROBE = "_patch_utils_behavior_probe"
ORIGINAL_SOURCE = f"""
calls = []

def {PROBE}(value, *extra, factor=1):
    calls.append((value, extra, factor))
    return value * factor
"""


@pytest.fixture
def patch_utils(monkeypatch):
    # Importing megatron_supa runs global adaptation. Load the real, dependency-free
    # leaf instead; each test gets its own PatchManager class and registry.
    source = Path(__file__).resolve().parents[2] / "megatron_supa/patchs/patch_utils.py"
    if not source.is_file():
        # CI can package tests beside an installed wheel without the repo source.
        package_spec = PathFinder.find_spec("megatron_supa")
        assert package_spec is not None and package_spec.submodule_search_locations
        source = Path(next(iter(package_spec.submodule_search_locations))) / "patchs/patch_utils.py"
    # Retain the package prefix for coverage's installed-wheel source matching.
    spec = importlib.util.spec_from_file_location("megatron_supa.patchs._patch_utils_under_test", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    # apply_patch scans all imported modules. Some modules lazily cache unknown
    # attributes (for example torch.ops); restore this probe name even there.
    missing = object()
    attributes = [
        (vars(imported), vars(imported).get(PROBE, missing))
        for imported in tuple(sys.modules.values())
        if isinstance(imported, ModuleType)
    ]
    try:
        yield module
    finally:
        for namespace, original in attributes:
            if original is missing:
                namespace.pop(PROBE, None)
            else:
                namespace[PROBE] = original


@pytest.fixture
def package(tmp_path, monkeypatch):
    name = f"_patch_utils_targets_{uuid4().hex}"
    directory = tmp_path / name
    directory.mkdir()
    (directory / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    def load(module_name, source=ORIGINAL_SOURCE):
        (directory / f"{module_name}.py").write_text(source, encoding="utf-8")
        importlib.invalidate_caches()
        return importlib.import_module(f"{name}.{module_name}")

    try:
        yield SimpleNamespace(name=name, load=load)
    finally:
        # Includes descendants created by Patch.parse_path, which monkeypatch
        # cannot track because production code writes sys.modules directly.
        for module_name in tuple(sys.modules):
            if module_name.startswith(name):
                del sys.modules[module_name]
        importlib.invalidate_caches()


def test_manager_defers_replacements_until_all_registered_targets_are_applied(patch_utils, package):
    first = package.load("first")
    second = package.load("second")

    def replacement(value, *, factor):
        return ("patched", value, factor)

    manager = patch_utils.PatchManager
    manager.register_patch(f"{first.__name__}.{PROBE}", replacement)
    manager.register_patch(f"{second.__name__}.{PROBE}", replacement)

    assert getattr(first, PROBE)(2, factor=3) == 6
    assert getattr(second, PROBE)(4, factor=5) == 20
    manager.apply_patches()
    assert getattr(first, PROBE)(2, factor=3) == ("patched", 2, 3)
    assert getattr(second, PROBE)(4, factor=5) == ("patched", 4, 5)


@pytest.mark.parametrize("already_applied", [False, True], ids=["pending", "applied"])
def test_duplicate_replacement_is_rejected_without_losing_first_patch(patch_utils, package, already_applied):
    target = package.load("target")
    path = f"{target.__name__}.{PROBE}"
    manager = patch_utils.PatchManager

    def first(value):
        return value + 10

    def conflicting(value):
        return value + 100

    manager.register_patch(path, first)
    if already_applied:
        manager.apply_patches()
    with pytest.raises(RuntimeError, match="patch.*exist"):
        manager.register_patch(path, conflicting)
    manager.apply_patches()
    assert getattr(target, PROBE)(2) == 12


@pytest.mark.parametrize("already_applied", [False, True], ids=["pending", "applied"])
def test_force_replacement_takes_effect_on_next_apply(patch_utils, package, already_applied):
    target = package.load("target")
    path = f"{target.__name__}.{PROBE}"
    manager = patch_utils.PatchManager

    def first(value):
        return value + 10

    def replacement(value):
        return value + 100

    manager.register_patch(path, first)
    if already_applied:
        manager.apply_patches()
    manager.register_patch(path, replacement, force_patch=True)
    assert getattr(target, PROBE)(2) == (12 if already_applied else 2)
    manager.apply_patches()
    assert getattr(target, PROBE)(2) == 102


@pytest.mark.parametrize("replace_original, expected", [(False, 55), (True, 60)])
def test_wrapper_and_decorator_compose_in_registration_order(patch_utils, package, replace_original, expected):
    target = package.load("target")
    path = f"{target.__name__}.{PROBE}"
    manager = patch_utils.PatchManager
    trace = []
    token = object()

    def replacement(value, *extra, factor=1):
        target.calls.append((value, extra, factor))
        return value * factor + 1

    def add_wrapper(function):
        def wrapped(*args, **kwargs):
            trace.append("add-enter")
            result = function(*args, **kwargs) + 3
            trace.append("add-exit")
            return result

        return wrapped

    def multiply_decorator(function):
        def wrapped(*args, **kwargs):
            trace.append("multiply-enter")
            result = function(*args, **kwargs) * 5
            trace.append("multiply-exit")
            return result

        return wrapped

    if replace_original:
        manager.register_patch(path, replacement)
    manager.register_patch(path, add_wrapper)
    manager.register_patch(path, multiply_decorator)
    manager.apply_patches()

    assert getattr(target, PROBE)(2, token, factor=4) == expected
    assert target.calls == [(2, (token,), 4)]
    assert target.calls[0][1][0] is token
    assert trace == ["multiply-enter", "add-enter", "add-exit", "multiply-exit"]


def test_wrapper_preserves_argument_and_return_object_identity(patch_utils, package):
    target = package.load(
        "target",
        f"""
calls = []

def {PROBE}(value, *extra, result):
    calls.append((value, extra))
    return result
""",
    )
    payload, extra, result = object(), object(), object()

    def passthrough_wrapper(function):
        def wrapped(*args, **kwargs):
            return function(*args, **kwargs)

        return wrapped

    patch_utils.PatchManager.register_patch(f"{target.__name__}.{PROBE}", passthrough_wrapper)
    patch_utils.PatchManager.apply_patches()

    assert getattr(target, PROBE)(payload, extra, result=result) is result
    assert target.calls == [(payload, (extra,))]
    assert target.calls[0][0] is payload
    assert target.calls[0][1][0] is extra


def test_repeated_apply_keeps_the_same_callable_and_does_not_wrap_again(patch_utils, package):
    target = package.load("target")
    constructions = []

    def increment_wrapper(function):
        constructions.append(function)

        def wrapped(value):
            return function(value) + 1

        return wrapped

    patch = patch_utils.Patch(f"{target.__name__}.{PROBE}", increment_wrapper, create_dummy=False)
    patch.apply_patch()
    applied = getattr(target, PROBE)
    assert applied(2) == 3
    patch.apply_patch()
    assert getattr(target, PROBE) is applied
    assert getattr(target, PROBE)(2) == 3
    assert len(constructions) == 1
    assert target.calls == [(2, (), 1), (2, (), 1)]


@pytest.mark.parametrize(
    "replace_original, first_result, second_result, third_result",
    [(False, 6, 12, 9), (True, 16, 32, 29)],
    ids=["original-with-wrappers", "replacement-with-wrappers"],
)
def test_wrappers_registered_after_apply_are_each_composed_once(
    patch_utils, package, replace_original, first_result, second_result, third_result
):
    target = package.load("target")
    consumer = package.load("consumer", f"from .target import {PROBE}\n")
    path = f"{target.__name__}.{PROBE}"
    manager = patch_utils.PatchManager

    def replacement(value):
        return value + 10

    def add_wrapper(function):
        return lambda value: function(value) + 1

    def double_wrapper(function):
        return lambda value: function(value) * 2

    def subtract_decorator(function):
        return lambda value: function(value) - 3

    if replace_original:
        manager.register_patch(path, replacement)
    manager.register_patch(path, add_wrapper)
    manager.apply_patches()
    assert getattr(target, PROBE)(5) == first_result

    manager.register_patch(path, double_wrapper)
    manager.apply_patches()
    assert getattr(target, PROBE)(5) == second_result
    assert getattr(consumer, PROBE)(5) == second_result

    manager.register_patch(path, subtract_decorator)
    manager.apply_patches()
    assert getattr(target, PROBE)(5) == third_result
    assert getattr(consumer, PROBE)(5) == third_result
    composed = getattr(target, PROBE)
    manager.apply_patches()
    assert getattr(target, PROBE) is composed
    assert getattr(target, PROBE)(5) == third_result


@pytest.mark.parametrize("already_applied", [False, True], ids=["pending", "applied"])
def test_force_replacement_retains_each_registered_wrapper_once(patch_utils, package, already_applied):
    target = package.load("target")
    path = f"{target.__name__}.{PROBE}"
    manager = patch_utils.PatchManager

    def first(value):
        return value + 10

    def replacement(value):
        return value + 100

    def add_wrapper(function):
        return lambda value: function(value) + 1

    def double_wrapper(function):
        return lambda value: function(value) * 2

    manager.register_patch(path, first)
    manager.register_patch(path, add_wrapper)
    manager.register_patch(path, double_wrapper)
    if already_applied:
        manager.apply_patches()
        assert getattr(target, PROBE)(5) == 32

    manager.register_patch(path, replacement, force_patch=True)
    manager.apply_patches()
    assert getattr(target, PROBE)(5) == 212
    manager.apply_patches()
    assert getattr(target, PROBE)(5) == 212


def test_apply_repairs_preimported_same_name_alias_but_preserves_other_functions(patch_utils, package):
    target = package.load("target")
    consumer = package.load(
        "consumer",
        f"""
from .target import {PROBE}

def invoke(value):
    return {PROBE}(value, factor=3)
""",
    )
    unrelated = package.load("unrelated")
    original = getattr(target, PROBE)
    assert getattr(consumer, PROBE) is original

    def replacement(value, *, factor):
        return value + factor

    patch_utils.PatchManager.register_patch(f"{target.__name__}.{PROBE}", replacement)
    patch_utils.PatchManager.apply_patches()

    assert getattr(consumer, PROBE) is replacement
    assert consumer.invoke(2) == 5
    assert getattr(target, PROBE)(2, factor=3) == 5
    assert getattr(unrelated, PROBE)(2, factor=3) == 6
    assert target.calls == []


@pytest.mark.parametrize("create_dummy", [False, True])
def test_replacement_can_add_a_missing_function_to_an_existing_module(patch_utils, package, create_dummy):
    target = package.load("target", "")
    assert not hasattr(target, PROBE)

    def replacement(value):
        return value + 10

    patch_utils.PatchManager.register_patch(f"{target.__name__}.{PROBE}", replacement, create_dummy=create_dummy)
    patch_utils.PatchManager.apply_patches()
    assert getattr(target, PROBE)(2) == 12


def test_create_dummy_builds_importable_nested_modules_and_installs_replacement(patch_utils, package):
    path = f"{package.name}.optional.backend"

    def replacement(value, *, factor):
        return value + factor

    patch_utils.PatchManager.register_patch(f"{path}.{PROBE}", replacement, create_dummy=True)
    patch_utils.PatchManager.apply_patches()

    target = importlib.import_module(path)
    optional = importlib.import_module(f"{package.name}.optional")
    parent = importlib.import_module(package.name)
    assert parent.optional is optional
    assert optional.backend is target
    assert importlib.util.find_spec(path).name == path
    assert getattr(target, PROBE)(2, factor=3) == 5


@pytest.mark.parametrize("function_exists", [False, True], ids=["missing-function", "existing-function"])
def test_omitting_replacement_installs_a_callable_that_reports_the_missing_target(
    patch_utils, package, function_exists
):
    target = package.load("target", ORIGINAL_SOURCE if function_exists else "")
    path = f"{target.__name__}.{PROBE}"
    patch_utils.PatchManager.register_patch(path)
    patch_utils.PatchManager.apply_patches()

    with pytest.raises(RuntimeError, match="no exist") as caught:
        getattr(target, PROBE)(object(), option=object())
    assert path in str(caught.value)


def test_module_only_registration_can_create_an_importable_dummy_root(patch_utils, package):
    module_name = f"{package.name}_optional"
    patch_utils.PatchManager.register_patch(module_name, create_dummy=True)
    patch_utils.PatchManager.apply_patches()

    module = importlib.import_module(module_name)
    assert module.__name__ == module_name
    assert importlib.util.find_spec(module_name).name == module_name


def test_missing_module_without_dummy_raises_and_can_be_retried_after_it_becomes_available(patch_utils, package):
    module_name = f"{package.name}.not_yet_available"

    def replacement(value):
        return value + 10

    patch_utils.PatchManager.register_patch(f"{module_name}.{PROBE}", replacement)
    with pytest.raises(ModuleNotFoundError) as caught:
        patch_utils.PatchManager.apply_patches()
    assert module_name in str(caught.value)
    assert module_name not in sys.modules
    assert not hasattr(importlib.import_module(package.name), "not_yet_available")

    target = package.load("not_yet_available")
    patch_utils.PatchManager.apply_patches()
    assert getattr(target, PROBE)(2) == 12


def test_dotted_attribute_target_replaces_a_class_method(patch_utils, package):
    target = package.load(
        "target",
        f"""
class Service:
    def {PROBE}(self, value, *, factor):
        return value * factor
""",
    )
    service = target.Service()

    def replacement(self, value, *, factor):
        assert self is service
        return value + factor

    assert getattr(service, PROBE)(2, factor=3) == 6
    patch_utils.PatchManager.register_patch(f"{target.__name__}.Service.{PROBE}", replacement)
    patch_utils.PatchManager.apply_patches()
    assert getattr(service, PROBE)(2, factor=3) == 5


@pytest.mark.parametrize("create_dummy", [False, True])
def test_missing_class_method_requires_dummy_permission(patch_utils, package, create_dummy):
    target = package.load("target", "class Service:\n    pass\n")

    def replacement(self, value):
        return value + 10

    patch_utils.PatchManager.register_patch(
        f"{target.__name__}.Service.{PROBE}", replacement, create_dummy=create_dummy
    )
    if not create_dummy:
        with pytest.raises(RuntimeError, match="no exist"):
            patch_utils.PatchManager.apply_patches()
        assert not hasattr(target.Service, PROBE)
    else:
        patch_utils.PatchManager.apply_patches()
        assert getattr(target.Service(), PROBE)(2) == 12


def test_wrapper_construction_error_leaves_original_callable_available_for_retry(patch_utils, package):
    target = package.load("target")
    original = getattr(target, PROBE)
    failures = [ValueError("wrapper unavailable")]

    def increment_wrapper(function):
        return lambda value: function(value) + 1

    def retry_wrapper(function):
        if failures:
            raise failures.pop()

        def wrapped(value):
            return function(value) + 10

        return wrapped

    patch_utils.PatchManager.register_patch(f"{target.__name__}.{PROBE}", increment_wrapper)
    patch_utils.PatchManager.register_patch(f"{target.__name__}.{PROBE}", retry_wrapper)
    with pytest.raises(ValueError, match="wrapper unavailable"):
        patch_utils.PatchManager.apply_patches()
    assert getattr(target, PROBE) is original
    assert getattr(target, PROBE)(2) == 2

    patch_utils.PatchManager.apply_patches()
    assert getattr(target, PROBE)(2) == 13


def test_wrapped_function_preserves_the_original_exception(patch_utils, package):
    target = package.load(
        "target",
        f"""
error = ValueError("original failure")

def {PROBE}(value):
    raise error
""",
    )

    def passthrough_wrapper(function):
        def wrapped(*args, **kwargs):
            return function(*args, **kwargs)

        return wrapped

    patch_utils.PatchManager.register_patch(f"{target.__name__}.{PROBE}", passthrough_wrapper)
    patch_utils.PatchManager.apply_patches()
    with pytest.raises(ValueError) as caught:
        getattr(target, PROBE)(2)
    assert caught.value is target.error
