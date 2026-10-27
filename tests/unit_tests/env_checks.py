"""Importable test helpers (use in @pytest.mark.skipif, etc.).

Names defined only in conftest are not available in test modules; put shared
checks here so tests can import them.
"""

import importlib.util


def has_transformer_engine() -> bool:
    """Return True if the ``transformer_engine`` package is installed."""
    return importlib.util.find_spec("transformer_engine") is not None
