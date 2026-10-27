from functools import wraps

import torch


def patch_br_torch():
    def type_wrapper(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            res = fn(*args, **kwargs)
            if isinstance(res, str):
                res = res.replace('supa', 'cuda')
            return res

        return wrapper

    def backend_wrapper(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            res = fn(*args, **kwargs)
            if isinstance(res, str) and res == 'bccl':
                res = 'nccl'
            return res

        return wrapper

    torch.nn.parameter.Parameter.type = type_wrapper(torch.nn.parameter.Parameter.type)  # replace supa type to gpu type
    torch.Tensor.type = type_wrapper(torch.Tensor.type)  # replace supa type to gpu type
    torch.distributed.get_backend = backend_wrapper(torch.distributed.get_backend)
    torch.distributed.distributed_c10d.get_backend = torch.distributed.get_backend
