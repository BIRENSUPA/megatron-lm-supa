import inspect
import logging
import sys
from functools import wraps

import torch
from torch import Tensor
from torch.profiler import record_function

logger = logging.getLogger(__name__)

_PATCHED_MAPPING = {}
_JIT_FUSER_PATCHED = False
_PROFILER_TRACE_NAMING_PATCHED = False
_ORIGINAL_EXPORT_CHROME_TRACE = None

# Symbols whose drifted `__module__` makes automatic traversal miss the Python
# wrapper. Values keep the profiler name stable; only Python functions are wrapped.
_FORCE_PATCH_SYMBOLS = {
    (
        'transformer_engine.pytorch.cpp_extensions',
        'general_grouped_gemm',
    ): 'nonaten.transformer_engine.pytorch.cpp_extensions.gemm.general_grouped_gemm',
    (
        'transformer_engine.pytorch.cpp_extensions',
        'grouped_gemm',
    ): 'nonaten.transformer_engine.pytorch.cpp_extensions.gemm.grouped_gemm',
    (
        'transformer_engine.pytorch.cpp_extensions',
        'te_grouped_gemm',
    ): 'nonaten.transformer_engine.pytorch.cpp_extensions.gemm.te_grouped_gemm',
}


def patch_torch_profiler_trace_naming():
    global _ORIGINAL_EXPORT_CHROME_TRACE, _PROFILER_TRACE_NAMING_PATCHED

    if _PROFILER_TRACE_NAMING_PATCHED:
        return

    # SUPA wraps the constructor; instances dispatch through the underlying class.
    profile_type = inspect.unwrap(torch.profiler.profile)
    original_export_chrome_trace = profile_type.export_chrome_trace
    _ORIGINAL_EXPORT_CHROME_TRACE = original_export_chrome_trace

    @wraps(original_export_chrome_trace)
    def export_chrome_trace_wrapper(self, path):
        try:
            from megatron.core import mpu

            if isinstance(path, str) and (path.endswith('.pt.trace.json') or path.endswith('.json.gz')):
                path_parts = path.rsplit('/', 1)
                profile_dir = path_parts[0] if len(path_parts) == 2 else '.'
                compressed = path.endswith('.gz')
                suffix = '.pt.trace.json.gz' if compressed else '.pt.trace.json'
                trace_name = 'rank{}_dp{}_pp{}_tp{}_cp{}{}'.format(
                    torch.distributed.get_rank(),
                    mpu.get_data_parallel_rank(),
                    mpu.get_pipeline_model_parallel_rank(),
                    mpu.get_tensor_model_parallel_rank(),
                    mpu.get_context_parallel_rank(),
                    suffix,
                )
                path = f'{profile_dir}/{trace_name}'
        except Exception as e:
            logger.debug('[Patch] Failed to customize torch profiler trace name: %s', e)
        return original_export_chrome_trace(self, path)

    profile_type.export_chrome_trace = export_chrome_trace_wrapper
    torch.profiler.profile.export_chrome_trace = export_chrome_trace_wrapper
    _PROFILER_TRACE_NAMING_PATCHED = True


def get_tensor_info(tensor):
    """Safely get tensor shape, strides and dtype info."""
    info = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    try:
        strides = tensor.stride()
        info["strides"] = list(strides) if strides else None
    except (AttributeError, RuntimeError):
        info["strides"] = None
    return info


def get_tensorlist_info_recursive(value, prefix=""):
    """Recursively get tensor list info with nested shapes and strides.

    Args:
        value: A tensor, list/tuple of tensors, or nested list/tuple of tensors
        prefix: Prefix for indexing in nested structures

    Returns:
        dict: Flattened tensor info with indexed keys
    """
    result = {}

    if isinstance(value, Tensor):
        info = {"shape": list(value.shape), "dtype": str(value.dtype)}
        try:
            strides = value.stride()
            info["strides"] = list(strides) if strides else None
        except (AttributeError, RuntimeError):
            info["strides"] = None
        result[prefix if prefix else "0"] = info
    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            result[prefix if prefix else "empty"] = {"type": "EmptyList", "count": 0}
        else:
            for i, item in enumerate(value):
                item_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
                sub_result = get_tensorlist_info_recursive(item, item_prefix)
                result.update(sub_result)
    else:
        result[prefix if prefix else "value"] = {"type": type(value).__name__, "value": str(value)}

    return result


def get_param_info(value):
    """Get parameter info for any value (tensor, tensor list, or constant).

    For tensor lists (including nested), returns a flattened structure
    with indexed keys for each tensor.
    """
    if isinstance(value, Tensor):
        return get_tensor_info(value)
    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            return {"type": "EmptyList", "count": 0}
        has_tensor = any(isinstance(v, Tensor) or isinstance(v, (list, tuple)) for v in value)
        if has_tensor:
            flattened = get_tensorlist_info_recursive(value)
            return {"type": "TensorList", "tensors": flattened}
        else:
            return {"type": type(value).__name__, "value": str(value)}
    else:
        return {"type": type(value).__name__, "value": str(value)}


def format_profiler_name(function_name, params):
    """Format function name with params for profiler trace.

    Since record_function args parameter doesn't work as expected,
    we embed the param info directly in the name.
    """
    if not params:
        return function_name

    parts = []
    for key, info in params.items():
        if "type" in info:
            if info["type"] == "TensorList":
                tensors = info.get("tensors", {})
                tensor_parts = []
                for tensor_key, tensor_info in tensors.items():
                    if "type" in tensor_info:
                        tensor_parts.append(f"{tensor_key}={tensor_info['type']}")
                    else:
                        dtype_str = tensor_info['dtype']
                        c10_dtype = _dtype_to_c10(dtype_str)
                        strides = tensor_info.get('strides', []) or []
                        tensor_parts.append(f"{tensor_key}={tensor_info['shape']}_{c10_dtype}_strides={strides}")
                part = f"{key}=TensorList[{len(tensors)}]:{{{','.join(tensor_parts)}}}"
            elif info["type"] == "EmptyList":
                part = f"{key}=EmptyList"
            else:
                part = f"{key}={info['type']}:{info['value']}"
        else:
            dtype_str = info['dtype']
            c10_dtype = _dtype_to_c10(dtype_str)
            strides = info['strides'] if info['strides'] else []
            part = f"{key}={info['shape']}_{c10_dtype}_strides={strides}"
        parts.append(part)

    return f"{function_name}|{'|'.join(parts)}"


def _dtype_to_c10(dtype_str):
    """Convert torch dtype string to C10 dtype name."""
    if 'bfloat16' in dtype_str.lower():
        return "BFloat16"
    elif 'float16' in dtype_str.lower() or 'half' in dtype_str.lower():
        return "Half"
    elif 'float64' in dtype_str.lower() or 'double' in dtype_str.lower():
        return "Double"
    elif 'float32' in dtype_str.lower() or 'float' in dtype_str.lower():
        return "Float"
    elif 'int64' in dtype_str.lower():
        return "Long"
    elif 'int32' in dtype_str.lower():
        return "Int"
    elif 'bool' in dtype_str.lower():
        return "Bool"
    else:
        return dtype_str.replace('torch.', '')


def capture_params(args, kwargs, start_idx=0):
    """Capture all parameters (tensors and constants) from args and kwargs.

    Args:
        args: Positional arguments tuple
        kwargs: Keyword arguments dict
        start_idx: Starting index for positional args (used for class methods to skip 'self')

    Returns:
        dict: Parameter info dictionary
    """
    params = {}
    for i, arg in enumerate(args[start_idx:], start=start_idx):
        params[f"arg_{i}"] = get_param_info(arg)
    for key, value in kwargs.items():
        params[key] = get_param_info(value)
    return params


def patch_jit_fuser():
    """Patch megatron.core.jit.jit_fuser to add profiler record_function.

    This function wraps the original jit_fuser to add detailed profiler tracing
    without modifying the original megatron code.

    It also handles the case where jit_fuser has already been used to decorate
    functions - it will find and re-wrap those functions with the profiler.
    """
    global _JIT_FUSER_PATCHED

    if _JIT_FUSER_PATCHED:
        logger.debug("[Patch] jit_fuser already patched, skipping...")
        return True

    try:
        from megatron.core import jit as jit_module

        # Get jit_fuser from the original jit module
        base_jit_fuser = getattr(jit_module, 'jit_fuser', None)
        if base_jit_fuser is None:
            logger.debug("[Patch] jit_fuser not found in megatron.core.jit, skipping jit_fuser patch")
            return False

        def create_jit_fuser_wrapper(func=None, *compile_args, **compile_kwargs):
            def decorate(real_func):
                fused_func = base_jit_fuser(real_func, *compile_args, **compile_kwargs)

                # Get full module path for the function
                module_name = getattr(real_func, '__module__', 'unknown')
                func_name = getattr(real_func, '__name__', 'unknown')
                full_module_path = f"{module_name}.{func_name}"

                @wraps(real_func)
                def wrapped_func(*args, **kwargs):
                    params = capture_params(args, kwargs)
                    full_name = format_profiler_name(f"nonaten.{full_module_path}", params)
                    with torch.profiler.record_function(full_name):
                        return fused_func(*args, **kwargs)

                wrapped_func._original_func = real_func
                wrapped_func._is_jit_patched = True
                wrapped_func._jit_wrapped = True
                return wrapped_func

            # Mimic torch.compile's two-stage decorator contract so all forms work:
            #   @jit_fuser            ->  create_jit_fuser_wrapper(func)
            #   jit_fuser(fn)          ->  create_jit_fuser_wrapper(fn)
            #   jit_fuser(fn, ...)     ->  create_jit_fuser_wrapper(fn, ...)
            #   jit_fuser(...)(fn)     ->  create_jit_fuser_wrapper(...)(fn)
            if func is None:
                return decorate
            return decorate(func)

        # Add to _PATCHED_MAPPING for update_imported_references to work
        _PATCHED_MAPPING[base_jit_fuser] = create_jit_fuser_wrapper
        jit_module.jit_fuser = create_jit_fuser_wrapper

        # Now find and patch all functions that were already decorated with jit_fuser
        patched_count = _patch_already_jit_fused_functions(base_jit_fuser)

        _JIT_FUSER_PATCHED = True
        logger.info(
            "[Patch] Successfully patched megatron.core.jit.jit_fuser (re-wrapped %s already-decorated functions)",
            patched_count,
        )
        return True

    except ImportError as e:
        logger.debug("[Patch] megatron.core.jit not available, skipping jit_fuser patch: %s", e)
        return False
    except Exception as e:
        logger.warning("[Patch] Failed to patch jit_fuser: %s", e)
        return False


def _is_jitted_function(func, base_jit_fuser):
    """Check if a function has been processed by jit_fuser.

    JIT-compiled functions (torch.jit.script or torch.compile) have specific attributes.
    """
    # Check for torch.jit.ScriptFunction
    if hasattr(func, 'forward') and hasattr(func, 'code'):
        return True
    # Check for torch.compile wrapped functions
    if hasattr(func, '_torchdynamo_orig_callable'):
        return True
    # Check if it's already been wrapped by us
    if hasattr(func, '_jit_wrapped'):
        return True
    return False


def _patch_already_jit_fused_functions(base_jit_fuser):
    """Find and re-wrap functions that were already decorated with jit_fuser.

    This function traverses all imported modules to find functions that have
    been processed by the original jit_fuser, and re-wraps them with our
    profiler wrapper.

    Args:
        base_jit_fuser: The original jit_fuser function (torch.jit.script or torch.compile)

    Returns:
        int: Number of functions that were re-wrapped
    """
    patched_count = 0
    visited_funcs = set()  # Track already processed functions to avoid duplicates

    # Get the type of torch.jit.ScriptFunction for isinstance check
    try:
        import torch.jit

        # script_function_type = type(torch.jit.script(lambda: None))
        script_function_type = type(None)
    except Exception:
        script_function_type = type(None)

    for module_name, module in list(sys.modules.items()):
        if module is None:
            continue

        if not hasattr(module, '__dict__'):
            continue

        # Skip certain modules that shouldn't be touched
        if any(skip in module_name for skip in ['torch.', 'torch_', 'sys', 'builtins', 'importlib']):
            continue

        try:
            for attr_name in list(module.__dict__.keys()):
                if attr_name.startswith('_'):
                    continue

                try:
                    attr_value = getattr(module, attr_name)
                except Exception:
                    continue

                # Skip if not callable or is a class/module
                if not callable(attr_value) or inspect.isclass(attr_value) or inspect.ismodule(attr_value):
                    continue

                # Use id to track unique functions
                func_id = id(attr_value)
                if func_id in visited_funcs:
                    continue
                visited_funcs.add(func_id)

                # Skip if already patched by us
                if hasattr(attr_value, '_is_jit_patched'):
                    continue

                # Check if this looks like a JIT-compiled function
                is_jitted = False
                original_func = None

                # Check 1: torch.jit.ScriptFunction
                if isinstance(attr_value, script_function_type):
                    is_jitted = True
                    # Try to get original function from the script function
                    original_func = getattr(attr_value, '_original_func', None)

                # Check 2: torch.compile wrapped functions (dynamo)
                elif hasattr(attr_value, '_torchdynamo_orig_callable'):
                    is_jitted = True
                    original_func = attr_value._torchdynamo_orig_callable

                # Check 3: Functions with code attribute that look like ScriptFunction
                elif hasattr(attr_value, 'code') and hasattr(attr_value, 'graph'):
                    is_jitted = True

                # Check 4: Look for functions that have been wrapped but don't have obvious JIT markers
                # We check by looking at the function's __wrapped__ chain
                elif hasattr(attr_value, '__wrapped__'):
                    # Check if the chain leads to something that looks JIT compiled
                    wrapped = attr_value
                    while hasattr(wrapped, '__wrapped__'):
                        wrapped = wrapped.__wrapped__
                        if isinstance(wrapped, script_function_type) or (
                            hasattr(wrapped, '_torchdynamo_orig_callable')
                        ):
                            is_jitted = True
                            original_func = wrapped
                            break

                # Check 5: Check if the function's __code__ or __globals__ indicate it's from jit_fuser
                # This is a heuristic based on the fact that jit_fuser returns torch.jit.script or torch.compile
                if not is_jitted:
                    # Check if function name suggests it's a fused/jitted function
                    func_name = getattr(attr_value, '__name__', '')
                    if func_name and ('fused' in func_name.lower() or 'jit' in func_name.lower()):
                        # Additional check: see if it's from megatron modules
                        func_module = getattr(attr_value, '__module__', '')
                        if 'megatron' in func_module:
                            is_jitted = True

                if is_jitted:
                    # Get the original function info for naming
                    if original_func is None:
                        original_func = attr_value

                    func_module = getattr(original_func, '__module__', module_name)
                    func_name = getattr(original_func, '__name__', attr_name)
                    full_module_path = f"{func_module}.{func_name}"

                    # Create a new wrapper that includes profiler
                    def create_profiler_wrapper(jitted_func, full_path):
                        def profiler_wrapper(*args, **kwargs):
                            params = capture_params(args, kwargs)
                            full_name = format_profiler_name(f"nonaten.{full_path}", params)
                            with torch.profiler.record_function(full_name):
                                return jitted_func(*args, **kwargs)

                        profiler_wrapper._original_func = getattr(jitted_func, '_original_func', jitted_func)
                        profiler_wrapper._is_jit_patched = True
                        profiler_wrapper._jit_wrapped = True
                        return profiler_wrapper

                    wrapped = create_profiler_wrapper(attr_value, full_module_path)

                    # Replace the function in the module
                    try:
                        setattr(module, attr_name, wrapped)
                        _PATCHED_MAPPING[attr_value] = wrapped
                        patched_count += 1
                    except Exception:
                        pass

        except Exception:
            # Skip modules that can't be inspected
            continue

    return patched_count


def create_wrapper(original_func, function_name):
    """Create a wrapper function that uses record_function to export parameters."""

    def wrapper(*args, **kwargs):
        params = capture_params(args, kwargs)
        full_name = format_profiler_name(function_name, params)
        with record_function(full_name):
            return original_func(*args, **kwargs)

    wrapper._original_func = original_func
    wrapper._wrapped_name = function_name
    _PATCHED_MAPPING[original_func] = wrapper
    return wrapper


def is_callable_function(obj, target_module='transformer_engine'):
    """Check if object is a callable function (not a class)."""
    return (
        callable(obj)
        and not inspect.isclass(obj)
        and not inspect.ismodule(obj)
        and hasattr(obj, '__module__')
        and obj.__module__
        and target_module in obj.__module__
    )


def traverse_and_patch_module(module, visited=None, prefix="", target_module='transformer_engine'):
    """Recursively traverse a module and patch all functions.

    Args:
        module: The module to traverse
        visited: Set of already visited modules to avoid cycles
        prefix: Prefix for the profiler name
        target_module: Target module name to filter (e.g., 'transformer_engine', 'apex')

    Returns:
        list: Names of patched functions
    """
    if visited is None:
        visited = set()

    module_id = id(module)
    if module_id in visited:
        return []
    visited.add(module_id)

    patched_functions = []

    try:
        for attr_name in dir(module):
            if attr_name.startswith('_'):
                continue

            try:
                attr = getattr(module, attr_name)
            except Exception:
                continue

            if inspect.ismodule(attr) and attr_name not in ['sys', 'os', 'torch', 'numpy']:
                if hasattr(attr, '__package__') and attr.__package__:
                    if target_module in attr.__package__ or target_module in attr.__name__:
                        sub_patched = traverse_and_patch_module(attr, visited, prefix, target_module)
                        patched_functions.extend(sub_patched)
                continue

            if not hasattr(attr, '__module__') or not attr.__module__:
                continue

            if target_module not in attr.__module__:
                continue

            func_name = attr.__name__ if hasattr(attr, '__name__') else attr_name
            full_name = f"{prefix}.{attr.__module__}.{func_name}" if prefix else f"{attr.__module__}.{func_name}"

            if is_callable_function(attr, target_module):
                try:
                    if hasattr(attr, '_is_patched'):
                        continue

                    wrapped = create_wrapper(attr, full_name)
                    wrapped._is_patched = True
                    wrapped._original_func = attr
                    setattr(module, attr_name, wrapped)
                    patched_functions.append(full_name)
                except Exception as e:
                    logger.warning("[Patch] Failed to patch function %s: %s", full_name, e)

    except Exception as e:
        logger.warning(
            "[Patch] Error traversing module %s: %s", module.__name__ if hasattr(module, '__name__') else 'unknown', e
        )

    return patched_functions


def update_imported_references():
    """Update references to patched functions in all already imported modules.

    When modules use 'from xxx import func', the function is bound directly
    to that module's namespace. This function traverses sys.modules to update
    those references to point to the wrapped versions.
    """
    if not _PATCHED_MAPPING:
        return

    updated_count = 0

    for module_name, module in list(sys.modules.items()):
        if module is None:
            continue

        if not hasattr(module, '__dict__'):
            continue

        try:
            for attr_name, attr_value in list(module.__dict__.items()):
                if attr_name.startswith('_'):
                    continue

                if attr_value in _PATCHED_MAPPING:
                    wrapped = _PATCHED_MAPPING[attr_value]
                    setattr(module, attr_name, wrapped)
                    updated_count += 1
                    logger.debug("[Patch] Updated reference: %s.%s -> wrapped version", module_name, attr_name)

        except Exception:
            pass

    if updated_count > 0:
        logger.info("[Patch] Updated %d function references in imported modules", updated_count)


def _force_patch_module_symbols(module, module_spec, prefix="nonaten"):
    """Force-wrap allowlisted Python-wrapper symbols whose runtime `__module__`
    no longer passes `traverse_and_patch_module`'s `target_module in __module__`
    filter (so auto-traversal skips them and the profiler top frame falls through
    to an inner pybind11 C function).

    For each `(module_spec, attr_name)` in `_FORCE_PATCH_SYMBOLS` resolving on
    this `module`, wrap the Python attr with `create_wrapper` using the
    dictionary's deterministic full_name (independent of the attr's drifted
    `__module__`), set the wrapper on the module, and register it in
    `_PATCHED_MAPPING` so `update_imported_references` re-points existing
    `from ... import general_grouped_gemm` bindings to the wrapped version.

    Guard: only wrap `inspect.isfunction` attrs (genuine Python wrappers).
    pybind11/builtin callables, `staticmethod`, and already-patched attrs are
    skipped — we never want to wrap the C entry point itself.

    Returns the list of full_names wrapped (empty if none applicable).
    """
    wrapped_names = []
    for (spec, attr_name), full_name in _FORCE_PATCH_SYMBOLS.items():
        if spec != module_spec:
            continue
        try:
            attr = getattr(module, attr_name, None)
        except Exception:
            attr = None
        if attr is None:
            continue
        # Only genuine Python functions — never wrap pybind11/builtins/staticmethod.
        if not inspect.isfunction(attr):
            continue
        if getattr(attr, '_is_patched', False) or getattr(attr, '_is_force_patched', False):
            continue
        # full_name already carries the `nonaten.` prefix literal from the allowlist.
        try:
            wrapped = create_wrapper(attr, full_name)
            wrapped._is_patched = True
            wrapped._is_force_patched = True
            wrapped._original_func = attr
            setattr(module, attr_name, wrapped)
            wrapped_names.append(full_name)
            logger.info("[Patch] Force-patched allowlisted symbol %s.%s -> %s", module_spec, attr_name, full_name)
        except Exception as e:
            logger.warning("[Patch] Failed to force-patch %s.%s: %s", module_spec, attr_name, e)
    return wrapped_names


def patch_module(module_spec):
    """Patch a module with given specification.

    Args:
        module_spec: Module import specification (e.g., 'transformer_engine_torch')

    Returns:
        int: Number of patched functions
    """
    try:
        parts = module_spec.split('.')
        module = __import__(module_spec)
        for part in parts[1:]:
            module = getattr(module, part)

        prefix = "nonaten"
        target_module = parts[0]

        logger.debug("[Patch] Starting auto-traversal of %s...", module_spec)
        patched_functions = traverse_and_patch_module(module, prefix=prefix, target_module=target_module)

        # Force-patch allowlisted symbols that auto-traversal skipped (e.g.
        # general_grouped_gemm whose `__module__` drifted off `transformer_engine`
        # in newer TE), so the profiler top frame is the Python wrapper, not the
        # inner pybind11 fn. See _FORCE_PATCH_SYMBOLS docstring (DM-61).
        force_wrapped = _force_patch_module_symbols(module, module_spec, prefix=prefix)
        if force_wrapped:
            patched_functions.extend(force_wrapped)

        if patched_functions:
            logger.info("[Patch] Successfully patched %d functions from %s:", len(patched_functions), module_spec)
            for func_name in patched_functions:
                logger.info("  - %s", func_name)
        else:
            logger.debug("[Patch] No functions found to patch in %s", module_spec)

        return len(patched_functions)

    except ImportError as e:
        logger.debug("[Patch] %s not available: %s", module_spec, e)
        return 0


def _build_fla_params(args, kwargs, sig_params):
    """Build a params dict for FLA entrypoints using the TE-style
    `arg_N=[shape]_dtype_strides=[...]` positional encoding for **tensor**
    parameters, and `key=value` kwarg encoding for non-tensor parameters.

    Why: btas `OpInfoReformat._parse_event_args` only fills `Input Dims /
    Input type / Concrete Inputs` from `arg_N=` segments; any `key=value`
    segment (even if its value is a tensor) lands in `params_kwargs` and is
    invisible to `has_unknown_input_shape`, which then drops the case
    (`Input Dims=[]`). FLA causal_conv1d is called entirely with kwargs
    (`x=...|weight=...|bias=...`), so without re-encoding the case never
    reaches `FlaOpGenerator`.

    Strategy:
      1. Bind actual args+kwargs to the function signature (positional order
         taken from `sig_params`). Falls back to positional-then-kwargs order
         when no signature is available.
      2. Walk bound parameters in signature order; for each parameter whose
         actual value is a Tensor (or TensorList) emit `arg_{i}=...` via the
         normal `get_param_info` + `format_profiler_name` tensor branch — the
         key is the synthetic `arg_{i}` so btas parses it as a positional
         tensor. Non-tensor parameters keep their real `name=value` kwarg
         form (so `activation=str:silu`, `output_final_state=bool:False`,
         `bias=NoneType:None` survive as params_kwargs, exactly as A100 spec
         wants).
      3. `arg_{i}` indices are assigned only to tensor params, in increasing
         order, so they match the `tensor_vars[0..N]` slots `FlaOpGenerator`
         reads.
    """
    from inspect import Parameter

    # Bind actual arguments to formal parameters, preserving signature order.
    # Only **explicitly passed** parameters are emitted (defaults skipped) —
    # this keeps the event name lean (FLA signatures have ~13 params, most
    # unused) and matches what megatron actually passes; unpassed defaults
    # are irrelevant to Input Dims and would only bloat params_kwargs.
    bound = {}
    if sig_params:
        pos_iter = iter(args)
        consumed_positional = 0
        for p in sig_params:
            if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD):
                if consumed_positional < len(args):
                    bound[p.name] = next(pos_iter)
                    consumed_positional += 1
                elif p.name in kwargs:
                    bound[p.name] = kwargs[p.name]
                # else: unpassed — skip (do NOT emit the default).
            elif p.kind == Parameter.VAR_POSITIONAL:
                for extra in pos_iter:
                    bound.setdefault(p.name, [])
                    bound[p.name].append(extra)
                consumed_positional = len(args)
            elif p.kind == Parameter.KEYWORD_ONLY:
                if p.name in kwargs:
                    bound[p.name] = kwargs[p.name]
                # else: unpassed keyword-only — skip.
            # VAR_KEYWORD: remaining unspecified kwargs; handled below.
        # Any kwargs not consumed by named params stay as their own entries.
        consumed_names = set(p.name for p in sig_params)
        for k, v in kwargs.items():
            if k not in consumed_names:
                bound[k] = v
    else:
        # No signature: positional first, then kwargs. Predictable enough for
        # the positional call sites (l2norm / chunk_gated_delta_rule today).
        for i, a in enumerate(args):
            bound[f"arg_{i}"] = a
        bound.update(kwargs)

    params = {}
    arg_idx = 0
    # `self` is never present here (entrypoints are module-level functions).
    for name, value in bound.items():
        info = get_param_info(value)
        is_tensor_like = ("shape" in info and "dtype" in info) or info.get("type") in ("TensorList", "EmptyList")
        if is_tensor_like:
            # Positional-encode the tensor: rewrite its key to arg_{arg_idx} so
            # format_profiler_name emits `arg_N=[shape]_dtype_strides=...`,
            # which btas parses into Input Dims.
            params[f"arg_{arg_idx}"] = info
            arg_idx += 1
        else:
            # Non-tensor (None / bool / int / float / str / ...): keep the
            # real parameter name so it lands in params_kwargs as `key=value`.
            params[name] = info
    return params


def _fla_tensor_infos_in_order(params):
    """Extract the tensor info dicts (shape/dtype/strides) from a params dict
    built by `_build_fla_params`, in `arg_N` order.

    Used to re-encode the backward event with the forward **input** tensor
    shapes (not grad_output shapes), so the bwd case's Input Dims match the
    fwd inputs and `FlaOpGenerator` can emit a `out = <func>(...);
    out.backward(ones_like(out))` body.
    """
    infos = []
    for key in sorted(
        (k for k in params if isinstance(k, str) and k.startswith('arg_')),
        key=lambda s: int(s.split('_', 1)[1]),
    ):
        infos.append(params[key])
    return infos


class _FlaBwdTagger(torch.autograd.Function):
    """Tag-along identity `autograd.Function` that emits a
    `nonaten.fla.<module>.<func>.backward|arg_N=...` record_function event
    during the backward pass, WITHOUT touching fla's own autograd.Function.

    Why a tag-along Function, not wrapping fla's `Class.backward`:
    PyTorch generates a separate `_backward_cls` (e.g. `L2NormFunctionBackward`)
    whose `.forward` is bound to fla's `.backward`; that proxy lacks the runtime
    state set up by custom_fwd (`_fwd_used_autocast`, `_dtype`), so wrapping
    `Class.backward` directly breaks AMP (`AttributeError:
    'L2NormFunctionBackward' object has no attribute '_fwd_used_autocast'`).
    Attaching this separate identity Function to the fwd output instead dodges
    fla's internal dispatch entirely — fla's backward still runs unchanged
    (our backward returns the grads as-is, then autograd proceeds to fla's
    grad_fn).

    fwd  `forward(ctx, bwd_name, input_infos, *fla_outputs)`:
        passes fla outputs through unchanged (numerically transparent; no
        custom_fwd, so AMP autocast state of fla's own Function is undisturbed).
    bwd  `backward(ctx, *grad_outputs)`:
        runs during `loss.backward()`, opens a record_function context tagged
        `bwd_name` + the forward input tensor shapes (encoded as
        `arg_N=[shape]_dtype_strides=[...]` via format_profiler_name — the
        TE-style positional pattern, so btas fills Input Dims and the bwd case
        is not dropped by `has_unknown_input_shape`), then returns the grads
        unchanged. `None` is returned for the two non-tensor leading inputs.

    Trade-off: our record_function opens/closes inside this Function's
    backward, so the `*.backward` user_annotation event is a sibling of fla's
    bwd kernels in the trace (not their parent) — wrapping fla's backward to
    nest them is the forbidden path. A sibling tag is sufficient for
    `FlaOpGenerator` to see and claim the bwd case.
    """

    @staticmethod
    def forward(ctx, bwd_name, input_infos, *outputs):
        ctx.bwd_name = bwd_name
        ctx.input_infos = input_infos
        return outputs if len(outputs) != 1 else outputs[0]

    @staticmethod
    def backward(ctx, *grad_outputs):
        params = {f"arg_{i}": info for i, info in enumerate(ctx.input_infos)}
        name = format_profiler_name(ctx.bwd_name, params)
        with record_function(name):
            pass
        # None for bwd_name / input_infos (non-tensor leads), then grads.
        return (None, None) + grad_outputs


# FLA bwd-kernel 名表（新分支：bwd case 直调 bwd kernel，不走 `.backward()`）。
#
# 每个 fwd entrypoint 映射到它 autograd `.backward` 内部真正调用的 bwd 入口函数
# （已读 fla 源码核实，READ-ONLY 未改 fla）：
#   - l2norm: L2NormFunction.backward -> l2norm_bwd(y, rstd, dy, eps)
#   - chunk_gated_delta_rule: ChunkGatedDeltaRuleFunction.backward ->
#     chunk_gated_delta_rule_bwd(q,k,v,g,beta,A,scale,ist,do,dht,...)
#   - causal_conv1d: CausalConv1dFunction.backward ->
#     causal_conv1d_bwd(x, dy, dht, weight, bias, ...)
# patcher 在 fla `.backward` 触发时把 record_function 事件名直接记成
# `nonaten.fla.<bwd_module>.<bwd_func>`（不再 `<func>.backward`），btas 剥 `nonaten.`
# 后 generate_cases 看到 `fla.fla.<bwd_module>.<bwd_func>` —— op_name 本身即 bwd
# kernel 名，FlaOpGenerator 据此路由到「直调 bwd kernel」分支（无 `.backward()`）。
#
# arg_N 仍编 fwd 输入 tensor 的 shape/dtype/strides（与 fwd 编码一致）——bwd body 内部
# 由 fwd 输入合成 y/rstd/A 等中间量 + grad（torch.ones_like），故 Input Dims 非
# 空/非 unknown，不被 `has_unknown_input_shape` 滤；tensor 槽位与 bwd body 消费一致。
#
# needs_fwd_intermediate: bwd kernel 是否需要 fwd 先跑出的中间量。
#   l2norm / chunk 需要（y,rstd / A）；causal_conv1d 在 activation=None 时 bwd 直吃 x，
#   不需要先跑 fwd（bwd 内部仅在 activation!=None 时自重算 y）。
#
# chunk_gated_delta_rule_refactored 在当前 fla 源码中不存在（前向兼容占位），无 bwd
# 映射；patcher import 失败即跳过，不进本表。
_FLA_BWD_KERNEL_FUNCS = {
    # fwd_func_name: (bwd_module, bwd_func, fwd_module, fwd_func, needs_fwd_intermediate)
    'l2norm': ('fla.modules.l2norm', 'l2norm_bwd', 'fla.modules.l2norm', 'l2norm_fwd', True),
    'chunk_gated_delta_rule': (
        'fla.ops.gated_delta_rule.chunk',
        'chunk_gated_delta_rule_bwd',
        'fla.ops.gated_delta_rule.chunk',
        'chunk_gated_delta_rule_fwd',
        True,
    ),
    'causal_conv1d': (
        'fla.modules.conv.triton',
        'causal_conv1d_bwd',
        'fla.modules.conv.triton',
        'causal_conv1d_fwd',
        False,
    ),
}


def _patch_fla_autograd_functions():
    """Add shape-aware trace events around supported FLA entrypoints.

    Public functions are wrapped instead of FLA autograd methods so AMP state
    and the original backward implementation remain untouched. Tensor inputs
    use ``arg_N`` encoding because the downstream parser ignores tensor kwargs;
    ``_FlaBwdTagger`` adds a transparent backward event with the forward input
    shapes for case generation.
    """
    try:
        from inspect import Parameter, signature
    except Exception:  # pragma: no cover
        signature = None
        Parameter = None

    targets = [
        ('fla.modules.l2norm', 'l2norm'),
        ('fla.ops.gated_delta_rule', 'chunk_gated_delta_rule'),
        ('fla.ops.gated_delta_rule', 'chunk_gated_delta_rule_refactored'),
        ('fla.modules.conv.causal_conv1d', 'causal_conv1d'),
    ]
    patched = 0
    for module_path, func_name in targets:
        try:
            mod = __import__(module_path, fromlist=[func_name])
            func = getattr(mod, func_name, None)
            if func is None or getattr(func, '_is_fla_patched', False):
                continue
            full_name = f"nonaten.fla.{module_path}.{func_name}"

            # Resolve the signature once to preserve formal parameter order.
            try:
                sig = signature(func) if signature is not None else None
            except (TypeError, ValueError):
                sig = None
            sig_params = list(sig.parameters.values()) if sig is not None else []

            # Capture loop values in defaults so each wrapper keeps its target.
            bwd_meta = _FLA_BWD_KERNEL_FUNCS.get(func_name)
            if bwd_meta is not None:
                bwd_full_name = f"nonaten.fla.{bwd_meta[0]}.{bwd_meta[1]}"
            else:
                bwd_full_name = f"{full_name}.backward"

            @wraps(func)
            def wrapped(
                *args, _func=func, _full_name=full_name, _bwd_full_name=bwd_full_name, _sig_params=sig_params, **kwargs
            ):
                params = _build_fla_params(args, kwargs, _sig_params)
                name = format_profiler_name(_full_name, params)
                input_infos = _fla_tensor_infos_in_order(params)
                with record_function(name):
                    outputs = _func(*args, **kwargs)
                # 在 autograd graph 上挂一层 identity autograd.Function，让 bwd 时由它
                # 打 `.backward` 标签事件（见 _FlaBwdTagger）。只接受 tensor 输出；标量 /
                # final_state 跳过。fla 输出约定：l2norm -> Tensor；chunk -> (o, state)；
                # causal_conv1d -> (output, final_state)。
                outs = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
                tensor_outs = tuple(o for o in outs if isinstance(o, Tensor))
                if tensor_outs:
                    tagged = _FlaBwdTagger.apply(_bwd_full_name, input_infos, *tensor_outs)
                    if len(tensor_outs) == 1 and isinstance(outputs, (tuple, list)):
                        new_outs = list(outputs)
                        for i, o in enumerate(new_outs):
                            if isinstance(o, Tensor):
                                new_outs[i] = tagged
                                break
                        outputs = type(outputs)(new_outs)
                    elif len(tensor_outs) == 1:
                        outputs = tagged
                    else:
                        new_outs = list(outputs) if isinstance(outputs, list) else list(outputs)
                        tag_iter = iter(tagged if isinstance(tagged, tuple) else (tagged,))
                        for i, o in enumerate(new_outs):
                            if isinstance(o, Tensor):
                                new_outs[i] = next(tag_iter)
                        outputs = type(outputs)(new_outs)
                return outputs

            wrapped._is_fla_patched = True
            wrapped._original_func = func
            setattr(mod, func_name, wrapped)
            _PATCHED_MAPPING[func] = wrapped
            patched += 1
        except Exception as e:
            logger.debug("[Patch] FLA %s.%s not patchable: %s", module_path, func_name, e)
    if patched:
        logger.info("[Patch] Patched %d FLA entrypoints (l2norm / chunk_gated_delta_rule / causal_conv1d)", patched)
    return patched


def apply_patches():
    """Apply patches to specified functions."""

    patch_torch_profiler_trace_naming()
    patch_jit_fuser()

    module_specs = [
        'transformer_engine_torch',
        'transformer_engine.pytorch.cpp_extensions',
        'apex.transformer.functional',
        'flash_attn',
        'fused_rotary_positional_embedding',
        'fused_weight_gradient_mlp_cuda',
    ]

    total_patched = 0
    for module_spec in module_specs:
        total_patched += patch_module(module_spec)

    # FLA entrypoints (l2norm / chunk_gated_delta_rule) are imported by
    # megatron.core.ssm.gated_delta_net. They bypass jit_fuser and the
    # module_specs list, so wrap the public functions explicitly to embed
    # input shapes in trace event names. See _patch_fla_autograd_functions.
    total_patched += _patch_fla_autograd_functions()

    update_imported_references()

    logger.info("[Patch] Total patched functions: %d", total_patched)
