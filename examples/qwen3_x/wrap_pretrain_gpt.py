"""Wrap pretrain_gpt.py so that megatron_supa patcher is imported BEFORE megatron.
Usage: torchrun [torchrun-args] <repo>/examples/tencent/wrap_pretrain_gpt.py [pretrain_gpt.py args...]
"""
import sys, os, runpy

# Resolve repo root from this file's location (examples/tencent/ -> ../..),
# so the wrapper is portable and never hard-codes an absolute path.
_HERE = os.path.dirname(os.path.abspath(__file__))
SUPA_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
TP_ROOT = os.path.join(SUPA_ROOT, "third-party", "megatron-lm")

sys.path.insert(0, SUPA_ROOT)
sys.path.insert(0, TP_ROOT)

# Activate supa patcher (imports fla entrypoints, wraps them, calls
# update_imported_references later). Must happen before `import megatron`.
import megatron_supa  # noqa: F401

os.chdir(TP_ROOT)
runpy.run_path(os.path.join(TP_ROOT, "pretrain_gpt.py"), run_name="__main__")
