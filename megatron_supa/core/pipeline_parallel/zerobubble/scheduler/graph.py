from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional


class FuncType(Enum):
    F = auto()
    B = auto()
    W = auto()
    SEND_FORWARD = auto()
    RECV_FORWARD = auto()
    SEND_BACKWARD = auto()
    RECV_BACKWARD = auto()


@dataclass(frozen=True)
class ScheduledNode:
    type: FuncType
    microbatch: int
    stage: Optional[int] = None
    layer_group_idx: Optional[int] = None


@dataclass
class GraphConfig:
    mem_f: List[float] = None
    mem_b: List[float] = None
    mem_w: List[float] = None
    max_mem: Optional[List[float]] = None
    cost_f: List[float] = None
    cost_b: List[float] = None
    cost_w: List[float] = None
    cost_comm: float = 0.0
    print_scaling: int = 1
    max_chunks: int = 1
    n_stages: int = None
    n_micro: int = None

    def __post_init__(self):
        assert all([isinstance(cost_f, float) for cost_f in self.cost_f])
        assert all([isinstance(cost_b, float) for cost_b in self.cost_b])
        assert all([isinstance(cost_w, float) for cost_w in self.cost_w])
        assert isinstance(self.cost_comm, float)
        assert all([f + b + w == 0 for (f, b, w) in zip(self.mem_f, self.mem_b, self.mem_w)])
        assert self.n_stages is not None
        assert self.n_micro is not None
