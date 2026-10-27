
import os
import torch

try:
    import torch_supa
    from torch_supa.contrib import transfer_to_supa
    PYTORCH_SUPA_INSTALL_PATH = os.path.dirname(os.path.realpath(torch_supa.__file__))

except ModuleNotFoundError:
    PYTORCH_SUPA_INSTALL_PATH = ""

from megatron.core import nccl_allocator 

def _build_nccl_allocator():
    # If the allocator is already built, return
    if nccl_allocator._allocator is not None:
        return

    nccl_allocator_source = """
    #include "torch_supa/csrc/core/supa/SUPACachingAllocator.h"
    #include <c10/util/Exception.h>
    #include "torch_supa/csrc/supa/SUPAPluggableAllocator.h"
    #include <torch/extension.h>
    
    #include <bccl.h>
    #include <iostream>
    #include <cstdio>
    
    extern "C" {
        #define NCCL_CHECK(cmd) do { \
        bcclResult_t r = cmd; \
        if (r != bcclSuccess) { \
            printf("Failed, NCCL error %s:%d '%s':", \
                __FILE__,__LINE__,bcclGetErrorString(r)); \
            exit(EXIT_FAILURE); \
        } \
        } while(0)
    
    
        C10_EXPORT  void* nccl_alloc_plug(size_t size, int device, void* stream) {
            void* ptr;
            NCCL_CHECK(bcclMemAlloc(&ptr, size));
            return ptr;
        }
    
        C10_EXPORT void nccl_free_plug(void* ptr, size_t size, int device, void* stream) {
            NCCL_CHECK(bcclMemFree(ptr));
        }
    
        std::shared_ptr<c10::supa::SUPACachingAllocator::SUPAAllocator> nccl_allocator;
    
        C10_EXPORT void maybe_init() {
            if (!nccl_allocator) {
                nccl_allocator = std::make_shared<
                    torch_supa::supa::SUPAPluggableAllocator::SUPAPluggableAllocator>(
                    nccl_alloc_plug, nccl_free_plug);
            }
        }
    
        C10_EXPORT std::shared_ptr<c10::supa::SUPACachingAllocator::SUPAAllocator>
        get_nccl_allocator() {
        maybe_init();
        return nccl_allocator;
        }
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("get_nccl_allocator", []() { return get_nccl_allocator(); });
        };
    }
    """
    module_dir = os.path.dirname(__file__)
    # build_directory: prefer a user-writable cache so this can be JIT-built by
    # non-root processes (e.g. CI running as a non-root user against a root-owned
    # installed package). Honor TORCH_EXTENSIONS_DIR if set, else fall back to the
    # standard torch extensions cache (~/.cache/torch_extensions). Only fall back
    # to the in-package `build/` dir (legacy behaviour, root-writable in source
    # trees) when neither cache path is usable.
    nccl_allocator_libname = "nccl_allocator"
    cache_root = os.environ.get("TORCH_EXTENSIONS_DIR")
    if cache_root:
        source_dir = os.path.join(cache_root, nccl_allocator_libname)
    else:
        source_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "torch_extensions", nccl_allocator_libname
        )
    try:
        os.makedirs(source_dir, exist_ok=True)
        if not os.access(source_dir, os.W_OK):
            raise PermissionError(source_dir)
    except (PermissionError, OSError):
        # Last resort: the in-package build dir (works for root / source-tree runs).
        source_dir = os.path.join(module_dir, "build")
        os.makedirs(source_dir, exist_ok=True)
    # import suda
    torch_supa_lib_path = os.path.join(PYTORCH_SUPA_INSTALL_PATH, "lib")
    nccl_allocator_extension = torch.utils.cpp_extension.load_inline(
        name=nccl_allocator_libname,
        cpp_sources=nccl_allocator_source,
        with_cuda=True,
        extra_ldflags=["-lbccl",f"-L{torch_supa_lib_path} -ltorch_supa"],
        verbose=True,
        is_python_module=True,
        build_directory=source_dir,
    )

    nccl_allocator._allocator = nccl_allocator_extension.get_nccl_allocator()
