import queue
from contextlib import contextmanager

from megatron.core import parallel_state
from megatron.training import get_args


class WeightGradStore:
    should_split_bw = False
    cache = []
    weight_grad_queue = None

    @classmethod
    def lazy_init(cls):
        if cls.weight_grad_queue is not None:
            return
        num_chunks = parallel_state.get_virtual_pipeline_model_parallel_world_size() or 1
        cls.weight_grad_queue = [[queue.Queue() for _ in range(1)] for _ in range(num_chunks)]

    @classmethod
    def is_supported(cls):
        args = get_args()
        if args.pipeline_model_parallel_size <= 1:
            return False
        if args.virtual_pipeline_model_parallel_size is not None:
            return False
        if args.overlap_grad_reduce:
            return False
        if not args.gradient_accumulation_fusion:
            return False
        if args.transformer_impl == 'transformer_engine':
            return False
        return True

    @classmethod
    def split_bw(cls):
        return cls.should_split_bw and cls.is_supported()

    @classmethod
    @contextmanager
    def set_split_bw(cls, enabled: bool):
        prev = cls.should_split_bw
        cls.should_split_bw = enabled
        try:
            yield
        finally:
            cls.should_split_bw = prev

    @classmethod
    def put(cls, weight, pre_func, func):
        assert cls.split_bw()
        cls.cache.append((weight, pre_func, func))

    @classmethod
    def flush(cls, chunk=0, seq_split_idx=0):
        cls.lazy_init()
        if not cls.split_bw():
            assert len(cls.cache) == 0
            return
        cls.weight_grad_queue[chunk][seq_split_idx].put(cls.cache)
        cls.cache = []

    @classmethod
    def pop(cls, chunk=0, seq_split_idx=0):
        cls.lazy_init()
        if cls.weight_grad_queue[chunk][seq_split_idx].empty():
            rank = parallel_state.get_pipeline_model_parallel_rank()
            raise RuntimeError(f'Pop empty weight grad queue on pipeline rank {rank}')
        stored_grads = cls.weight_grad_queue[chunk][seq_split_idx].get()
        for _, pre_func, func in stored_grads:
            func(*pre_func(async_op=False))

    @classmethod
    def clear(cls, model=None, chunk=0, seq_split_idx=0):
        cls.lazy_init()
        while not cls.weight_grad_queue[chunk][seq_split_idx].empty():
            cls.pop(chunk, seq_split_idx)

    @classmethod
    def assert_empty(cls):
        assert len(cls.cache) == 0, 'weight grad cache is not empty'
        if cls.weight_grad_queue is None:
            return
        for chunk_q in cls.weight_grad_queue:
            for seq_q in chunk_q:
                assert seq_q.empty(), 'weight grad queue is not empty'
