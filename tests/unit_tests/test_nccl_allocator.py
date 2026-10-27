"""Test the SUPA allocator builder and its native allocation/free bridge."""

import gc
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from megatron.core import nccl_allocator as core_allocator

from megatron_supa.core import nccl_allocator

pytestmark = [pytest.mark.sanity, pytest.mark.regression]


@pytest.fixture
def build_context(monkeypatch, tmp_path):
    # Keep real directory handling; only replace the external compiler/factory.
    allocator = object()
    factory = Mock(return_value=allocator)
    compiler = Mock(return_value=SimpleNamespace(get_nccl_allocator=factory))
    cache_root = tmp_path / "extensions"
    package_root = tmp_path / "torch_supa"
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(cache_root))
    monkeypatch.setattr(core_allocator, "_allocator", None)
    monkeypatch.setattr(nccl_allocator, "PYTORCH_SUPA_INSTALL_PATH", str(package_root))
    monkeypatch.setattr(torch.utils.cpp_extension, "load_inline", compiler)
    return SimpleNamespace(
        allocator=allocator,
        factory=factory,
        compiler=compiler,
        cache_dir=cache_root / "nccl_allocator",
        package_root=package_root,
    )


def test_core_uses_supa_allocator_builder():
    assert core_allocator._build_nccl_allocator is nccl_allocator._build_nccl_allocator


def test_build_links_bccl_and_installs_the_returned_allocator(build_context):
    nccl_allocator._build_nccl_allocator()

    assert core_allocator._allocator is build_context.allocator
    build_context.compiler.assert_called_once()
    kwargs = build_context.compiler.call_args.kwargs
    assert kwargs["extra_ldflags"] == ["-lbccl", f"-L{build_context.package_root}/lib -ltorch_supa"]
    assert kwargs["with_cuda"] is True
    assert kwargs["is_python_module"] is True
    assert kwargs["build_directory"] == str(build_context.cache_dir)
    assert build_context.cache_dir.is_dir()


@pytest.mark.parametrize("override", [None, ""], ids=["unset", "empty"])
def test_default_cache_is_used_without_an_override(build_context, monkeypatch, tmp_path, override):
    if override is None:
        monkeypatch.delenv("TORCH_EXTENSIONS_DIR")
    else:
        monkeypatch.setenv("TORCH_EXTENSIONS_DIR", override)
    user_dir = tmp_path / "user"
    expanduser = os.path.expanduser
    monkeypatch.setattr(os.path, "expanduser", lambda path: str(user_dir) if path == "~" else expanduser(path))

    nccl_allocator._build_nccl_allocator()

    expected = user_dir / ".cache" / "torch_extensions" / "nccl_allocator"
    assert expected.is_dir()
    assert build_context.compiler.call_args.kwargs["build_directory"] == str(expected)
    assert core_allocator._allocator is build_context.allocator


@pytest.mark.parametrize("error_type", [PermissionError, OSError])
def test_cache_creation_failure_uses_package_build_dir(build_context, monkeypatch, tmp_path, error_type):
    monkeypatch.setattr(nccl_allocator, "__file__", str(tmp_path / "nccl_allocator.py"))
    makedirs = os.makedirs

    def create_directory(path, exist_ok=False):
        if Path(path) == build_context.cache_dir:
            raise error_type("cache directory unavailable")
        return makedirs(path, exist_ok=exist_ok)

    monkeypatch.setattr(os, "makedirs", create_directory)

    nccl_allocator._build_nccl_allocator()

    expected = tmp_path / "build"
    assert expected.is_dir()
    assert build_context.compiler.call_args.kwargs["build_directory"] == str(expected)
    assert core_allocator._allocator is build_context.allocator


def test_unwritable_cache_uses_package_build_dir(build_context, monkeypatch, tmp_path):
    monkeypatch.setattr(nccl_allocator, "__file__", str(tmp_path / "nccl_allocator.py"))
    access = os.access
    monkeypatch.setattr(
        os,
        "access",
        lambda path, mode: False if Path(path) == build_context.cache_dir else access(path, mode),
    )

    nccl_allocator._build_nccl_allocator()

    expected = tmp_path / "build"
    assert expected.is_dir()
    assert build_context.compiler.call_args.kwargs["build_directory"] == str(expected)
    assert core_allocator._allocator is build_context.allocator


def test_unusable_cache_and_fallback_do_not_initialize_allocator(build_context, monkeypatch, tmp_path):
    monkeypatch.setattr(nccl_allocator, "__file__", str(tmp_path / "nccl_allocator.py"))
    makedirs = os.makedirs
    fallback = tmp_path / "build"

    def create_directory(path, exist_ok=False):
        if Path(path) in (build_context.cache_dir, fallback):
            raise PermissionError("both build directories unavailable")
        return makedirs(path, exist_ok=exist_ok)

    monkeypatch.setattr(os, "makedirs", create_directory)

    with pytest.raises(PermissionError, match="both build directories unavailable"):
        nccl_allocator._build_nccl_allocator()

    assert core_allocator._allocator is None
    build_context.compiler.assert_not_called()


@pytest.mark.parametrize("failure_stage", ["compiler", "factory"])
def test_failed_build_can_be_retried(build_context, failure_stage):
    failing_step = getattr(build_context, failure_stage)
    failing_step.side_effect = RuntimeError("allocator build failed")

    with pytest.raises(RuntimeError, match="allocator build failed"):
        nccl_allocator._build_nccl_allocator()

    assert core_allocator._allocator is None

    failing_step.side_effect = None
    nccl_allocator._build_nccl_allocator()

    assert core_allocator._allocator is build_context.allocator


def test_initialized_allocator_is_reused_without_touching_the_cache(build_context, monkeypatch):
    monkeypatch.setattr(core_allocator, "_allocator", build_context.allocator)

    nccl_allocator._build_nccl_allocator()

    assert core_allocator._allocator is build_context.allocator
    assert not build_context.cache_dir.exists()
    build_context.compiler.assert_not_called()


def test_native_allocator_compiles_allocates_and_reclaims_a_pool(monkeypatch, tmp_path):
    if not nccl_allocator.PYTORCH_SUPA_INSTALL_PATH or not torch.cuda.is_available():
        pytest.skip("Requires torch_supa and a SUPA device")

    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "extensions"))
    monkeypatch.setattr(core_allocator, "_allocator", None)
    core_allocator._build_nccl_allocator()
    assert core_allocator._allocator is not None

    # Normalize caches left by earlier tests before comparing reserved memory.
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    reserved_before = torch.cuda.memory_reserved()
    pool = torch.cuda.MemPool(core_allocator._allocator)
    tensor = None
    try:
        with torch.cuda.use_mem_pool(pool):
            tensor = torch.full((256,), 7, dtype=torch.int32, device="cuda")
        torch.cuda.synchronize()
        assert tensor.cpu().tolist() == [7] * 256
        assert pool.snapshot()
        assert torch.cuda.memory_reserved() > reserved_before
    finally:
        torch.cuda.synchronize()
        del tensor
        del pool
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    assert torch.cuda.memory_reserved() == reserved_before
