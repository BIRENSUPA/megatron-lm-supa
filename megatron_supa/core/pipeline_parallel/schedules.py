from functools import wraps

from megatron.training import get_args


def get_forward_backward_func_decorator(get_forward_backward_func):
    @wraps(get_forward_backward_func)
    def wrapper(pp_size=None, vp_size=None, schedule_pg_collection=None):
        try:
            args = get_args()
        except AssertionError:
            args = None

        if args is not None and getattr(args, 'enable_zero_bubble', False):
            from megatron_supa.core.pipeline_parallel.zerobubble import get_zero_bubble_forward_backward_func

            return get_zero_bubble_forward_backward_func()

        return get_forward_backward_func(
            pp_size=pp_size,
            vp_size=vp_size,
            schedule_pg_collection=schedule_pg_collection,
        )

    return wrapper
