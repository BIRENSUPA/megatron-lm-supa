from dataclasses import dataclass
from typing import List, Set

import torch

from .graph import FuncType, GraphConfig, ScheduledNode


@dataclass
class Graph:
    nstages: int
    nmb: int
    nnodes: int
    config: GraphConfig
    parents: List[Set[int]] = None
    name: List[str] = None
    precede: torch.Tensor = None

    # ID mapping:
    # F[stage][minibatch]: 0..STAGE* MB
    # B[stage][minibatch]: STAGE* MB .. 2 * STAGE * MB
    # W[stage][minibatch]: 2 * STAGE* MB .. 3 * STAGE * MB

    def get_id(self, type, stage, mb):
        return type * (self.nstages * self.nmb) + stage * self.nmb + mb

    def get_stage(self, id):
        return (id // self.nmb) % self.nstages

    def get_cost(self, id):
        type = id // (self.nstages * self.nmb)
        stage = self.get_stage(id)
        return [self.config.cost_f[stage], self.config.cost_b[stage], self.config.cost_w[stage]][type]

    def get_mem(self, id):
        type = id // (self.nstages * self.nmb)
        stage = self.get_stage(id)
        return [self.config.mem_f[stage], self.config.mem_b[stage], self.config.mem_w[stage]][type]

    def requires_order(self, i, j):
        return i != j and not self.precede[i][j] and not self.precede[j][i] and self.get_stage(i) == self.get_stage(j)

    @classmethod
    def build_graph(cls, nstages, nmb, config):
        nnodes = nstages * nmb * 3
        g = Graph(nstages=nstages, nmb=nmb, nnodes=nnodes, config=config)
        parents = []
        name = []
        for type in range(3):
            for stage in range(nstages):
                for mb in range(nmb):
                    p = set()
                    if type == 0:
                        name.append(f'F{mb}')
                        if stage > 0:
                            p.add(g.get_id(type, stage - 1, mb))
                        if mb > 0:
                            p.add(g.get_id(type, stage, mb - 1))
                    elif type == 1:
                        name.append(f'B{mb}')
                        if stage == nstages - 1:
                            p.add(g.get_id(0, stage, mb))
                        else:
                            p.add(g.get_id(type, stage + 1, mb))
                        if mb > 0:
                            p.add(g.get_id(type, stage, mb - 1))
                    elif type == 2:
                        name.append(f'W{mb}')
                        p.add(g.get_id(1, stage, mb))
                        if mb > 0:
                            p.add(g.get_id(type, stage, mb - 1))
                    else:
                        assert False
                    parents.append(p)

        g.name = name
        g.parents = parents
        return g

    # Manual ordering producing this kind of schedule:
    # fffffffbfbfbfbfbfbwbwbwbwbwbwbwwwwww
    #  fffffbfbfbfbfbfbfbfbwbwbwbwbwwwwwwww
    #   fffbfbfbfbfbfbfbfbfbfbwbwbwwwwwwwwww
    #    fbfbfbfbfbfbfbfbfbfbfbfbwwwwwwwwwwww
    # Returns the order index of each node on its own stage
    def manual_order(self, allow_bubble_before_first_b=False, prioritize_b=False, no_bubble_greedy=True):
        """Zero-Bubble流水线调度算法：生成无气泡的最优执行顺序

        该算法通过贪心策略和动态规划，在满足内存约束和通信依赖的前提下，
        最小化流水线训练时间，实现零气泡（bubble-free）的并行执行。

        Args:
            allow_bubble_before_first_b: 是否允许第一个反向传播前存在气泡
            prioritize_b: 是否优先调度反向传播以减少整体气泡
            no_bubble_greedy: 是否采用贪心策略尽可能消除气泡

        Returns:
            order: 每个节点在其stage内的执行顺序索引
            t: 每个节点的完成时间数组
            best_time: 整个流水线的总耗时（最慢stage的时间）
        """
        # ========== 状态变量初始化 ==========
        order = [0] * self.nnodes  # 每个节点在其stage内的执行顺序
        f = [0] * self.nstages  # 各stage已调度的F节点数
        b = [0] * self.nstages  # 各stage已调度的B节点数
        w = [0] * self.nstages  # 各stage已调度的W节点数
        o = [0] * self.nstages  # 各stage的总调度步数
        m = [0] * self.nstages  # 各stage当前内存占用
        e = [0] * self.nstages  # 各stage的预计结束时间
        t = [0] * self.nnodes  # 每个节点的实际完成时间

        # 内存限制：使用配置值或默认值（3倍微批次内存）
        max_mem = self.config.max_mem or [
            self.get_mem(self.get_id(0, stage, 0)) * self.nmb * 3 for stage in range(self.nstages)
        ]
        comm = self.config.cost_comm  # 通信延迟
        order_str = [""] * self.nstages  # 调试用：各stage的调度序列字符串
        stage_bubble = [0] * self.nstages  # 各stage累计气泡时间

        def get_max_bubble():
            """获取所有stage中的最大气泡时间"""
            max_bubble = 0
            for bb in stage_bubble:
                max_bubble = max(max_bubble, bb)
            return max_bubble

        def put(stage_j, type_k):
            if type_k == 0:
                _i = f[stage_j]
            elif type_k == 1:
                _i = b[stage_j]
            else:
                _i = w[stage_j]
            _j = stage_j
            _id = self.get_id(type_k, _j, _i)
            _mem = self.get_mem(_id)
            _cost = self.get_cost(_id)
            # TODO
            # assert m[_j] + _mem <= max_mem[stage_j]

            tmp = e[_j] + _cost
            no_bubble = tmp
            if _j > 0 and type_k == 0:
                tmp = max(tmp, t[self.get_id(0, _j - 1, _i)] + comm + _cost)
            if _j < self.nstages - 1 and type_k == 1:
                tmp = max(tmp, t[self.get_id(1, _j + 1, _i)] + comm + _cost)
            if f[_j] > 0:
                stage_bubble[_j] += tmp - no_bubble
            e[_j] = tmp
            t[_id] = tmp
            m[_j] += _mem
            order[_id] = o[_j]
            if type_k == 0:
                f[_j] += 1
            elif type_k == 1:
                b[_j] += 1
            else:
                w[_j] += 1
            o[_j] += 1
            fbw = "fbw"
            order_str[stage_j] += fbw[type_k]

        for i in range(self.nmb):
            if i == 0:
                for j in range(self.nstages):
                    put(j, 0)
                f_required = [0] * self.nstages
                last_t = 0
                for j in range(self.nstages - 1, -1, -1):
                    if j == self.nstages - 1:
                        last_t = t[self.get_id(0, j, i)] + self.get_cost(self.get_id(1, j, i))
                        continue
                    mem = m[j]
                    cost = e[j]
                    while True:
                        f_id = self.get_id(0, j, f[j] + f_required[j])
                        if f[j] + f_required[j] < self.nmb and mem + self.get_mem(f_id) <= max_mem[j]:
                            if allow_bubble_before_first_b:
                                if cost + self.get_cost(f_id) > last_t + comm:
                                    break
                            else:
                                if cost >= last_t + comm:
                                    break
                            mem += self.get_mem(f_id)
                            cost += self.get_cost(f_id)
                            f_required[j] += 1
                        else:
                            break
                    last_t = max(cost, last_t + comm) + self.get_cost(self.get_id(1, j, i))
                for j in range(self.nstages):
                    while (
                        j > 0
                        and f_required[j] > 0
                        and f_required[j] >= f_required[j - 1]
                        and f[j] + f_required[j] < self.nmb
                    ):
                        f_required[j] -= 1
                for j in range(self.nstages):
                    for _ in range(f_required[j]):
                        put(j, 0)
                for j in range(self.nstages - 1, -1, -1):
                    put(j, 1)
                continue
            f_required = [0] * self.nstages
            for j in range(self.nstages):
                if f[j] >= self.nmb:
                    continue
                if j + 1 < self.nstages and f[j] >= f[j + 1] + 2 and prioritize_b:
                    next_plus_fw = (
                        e[j + 1]
                        + self.get_cost(self.get_id(0, j + 1, f[j + 1]))
                        + self.get_cost(self.get_id(1, j + 1, b[j + 1]))
                        + comm
                    )
                    if e[j] >= next_plus_fw:
                        continue
                    f_id = self.get_id(0, j, f[j])
                    f_mem = self.get_mem(f_id)
                    w_cost, w_cnt = 0, 0
                    mem_with_w = m[j] + f_mem
                    while mem_with_w > max_mem[j] and w[j] + w_cnt < b[j]:
                        w_id = self.get_id(2, j, w[j] + w_cnt)
                        w_cost += self.get_cost(w_id)
                        mem_with_w += self.get_mem(w_id)
                        w_cnt += 1
                    if e[j] + self.get_cost(f_id) + w_cost <= next_plus_fw:
                        f_required[j] = 1
                        continue

                    w_cost, w_cnt = 0, 0
                    # mem_with_w = m[j]
                    # while w[j] + w_cnt < b[j]:
                    #     w_id = self.get_id(2, j, w[j] + w_cnt)
                    #     w_cost += self.get_cost(w_id)
                    #     mem_with_w += self.get_mem(w_id)
                    #     w_cnt += 1
                    # if e[j] + w_cost >= next_plus_fw:
                    #     continue
                    if next_plus_fw - (e[j] + w_cost) <= get_max_bubble() - stage_bubble[j]:
                        # TODO: can sample here
                        continue
                f_required[j] = 1
            for j in range(self.nstages - 2, -1, -1):
                f_required[j] = min(f_required[j], f_required[j + 1])
            for j in range(self.nstages):
                if f_required[j] == 0:
                    continue
                f_id = self.get_id(0, j, f[j])
                mem = self.get_mem(f_id)
                while m[j] + mem > max_mem[j]:
                    if w[j] >= b[j]:
                        raise ValueError("Cannot fit memory")
                    put(j, 2)
                if j > 0:
                    while (
                        w[j] < b[j]
                        and e[j] + self.get_cost(self.get_id(2, j, w[j])) <= t[self.get_id(0, j - 1, f[j])] + comm
                    ):
                        put(j, 2)
                    if w[j] < b[j] and e[j] < t[self.get_id(0, j - 1, f[j])] + comm:
                        # TODO: e[j] + self.get_cost(self.get_id(2, j, w[j])) > t[self.get_id(0, j - 1, f[j])] + comm
                        if t[self.get_id(0, j - 1, f[j])] + comm - e[j] <= get_max_bubble() - stage_bubble[j]:
                            # TODO: can sample here
                            if no_bubble_greedy:
                                put(j, 2)
                        else:
                            put(j, 2)
                put(j, 0)
            for j in range(self.nstages - 1, -1, -1):
                assert b[j] == i
                b_id = self.get_id(1, j, b[j])
                mem = self.get_mem(b_id)
                while m[j] + mem > max_mem[j]:
                    if w[j] >= b[j]:
                        raise ValueError("Cannot fit memory")
                    put(j, 2)
                if j + 1 < self.nstages:
                    while (
                        w[j] < b[j]
                        and e[j] + self.get_cost(self.get_id(2, j, w[j])) <= t[self.get_id(1, j + 1, i)] + comm
                    ):
                        put(j, 2)
                    if w[j] < b[j] and e[j] < t[self.get_id(1, j + 1, i)] + comm:
                        # TODO: e[j] + self.get_cost(self.get_id(2, j, w[j])) > t[self.get_id(1, j + 1, i)] + comm
                        if t[self.get_id(1, j + 1, i)] + comm - e[j] <= get_max_bubble() - stage_bubble[j]:
                            # TODO: can sample here
                            if no_bubble_greedy:
                                put(j, 2)
                        else:
                            put(j, 2)
                if j == 0 and f[j] == self.nmb:
                    while w[j] < b[j]:
                        put(j, 2)
                put(j, 1)

        for i in range(self.nstages):
            while w[i] < self.nmb:
                put(i, 2)
            # print(f"{' ' * i}{order_str[i]}  -> {e[i]}")

        for i in range(self.nstages):
            for j in range(self.nmb):
                f_id = self.get_id(0, i, j)
                b_id = self.get_id(1, i, j)
                w_id = self.get_id(2, i, j)
                f_cost = self.get_cost(f_id)
                b_cost = self.get_cost(b_id)
                w_cost = self.get_cost(w_id)
                assert t[b_id] >= t[f_id] + b_cost
                assert t[w_id] >= t[b_id] + w_cost, f"{i}-{j}, {t[w_id]} >= {t[b_id]} + {w_cost}"
                if i > 0:
                    assert t[f_id] >= t[self.get_id(0, i - 1, j)] + comm + f_cost, f"{i}-{j}"
                if i < self.nstages - 1:
                    assert t[b_id] >= t[self.get_id(1, i + 1, j)] + comm + b_cost

        # print(order)
        best_time = 0
        for i in range(self.nstages):
            time_i = t[self.get_id(2, i, self.nmb - 1)] - t[self.get_id(0, i, 0)] + self.get_cost(self.get_id(0, i, 0))
            best_time = max(best_time, time_i)

        return order, t, best_time


def initial_solution(graph, print_result=True):
    """通过网格搜索寻找最优调度方案

    遍历 `manual_order` 函数的所有参数组合（共8种），选择总耗时最短的调度方案。
    这是一种启发式搜索策略，通过穷举有限的参数空间来找到近似最优解。

    Args:
        graph: Graph对象，包含计算图结构和配置信息
        print_result: 是否打印调度结果详情

    Returns:
        best_time: 最优方案的总耗时
        order: 最优方案中每个节点的执行顺序
        complete_time: 最优方案中每个节点的完成时间
    """
    # 初始化最优解
    best_time, order, complete_time = None, None, None

    # 网格搜索：遍历所有8种参数组合
    # 参数空间：3个布尔参数，共 2^3 = 8 种组合
    for allow_bubble_before_first_b in [True, False]:
        for prioritize_b in [True, False]:
            for no_bubble_greedy in [True, False]:
                # 使用当前参数组合执行调度
                order_t, complete_time_t, best_time_t = graph.manual_order(
                    allow_bubble_before_first_b=allow_bubble_before_first_b,
                    prioritize_b=prioritize_b,
                    no_bubble_greedy=no_bubble_greedy,
                )

                # 更新最优解
                if best_time is None or best_time_t < best_time:
                    best_time = best_time_t
                    order = order_t
                    complete_time = complete_time_t

    # 可选：打印最优方案详情
    if print_result:
        print_detail(graph, complete_time)
        print("-" * 20, best_time, "-" * 20)

    return best_time, order, complete_time


def print_detail(graph, F):
    typenames = ['F', 'B', 'W']
    times = []
    for stage in range(graph.nstages):
        stage_str = ['.'] * int(F[graph.get_id(2, stage, graph.nmb - 1)] / graph.config.print_scaling)
        for _type in range(3):
            for _mb in range(graph.nmb):
                _id = graph.get_id(_type, stage, _mb)
                end = int(F[_id] / graph.config.print_scaling)
                start = int((F[_id] - graph.get_cost(_id)) / graph.config.print_scaling)
                for j in range(start, end):
                    if j == start or j == end - 1:
                        stage_str[j] = typenames[_type]
                    elif j == start + 1:
                        if _mb >= 10:
                            stage_str[j] = str(_mb // 10)
                        else:
                            stage_str[j] = str(_mb)
                    elif j == start + 2 and _mb >= 10:
                        stage_str[j] = str(_mb % 10)
                    else:
                        stage_str[j] = "-"
        _str = ""
        for _c in stage_str:
            _str += _c
        times.append(
            F[graph.get_id(2, stage, graph.nmb - 1)]
            - F[graph.get_id(0, stage, 0)]
            + graph.get_cost(graph.get_id(0, stage, 0))
        )
        print(_str)
    print('Longest stage time: ', max(times))


def create_schedule(config: GraphConfig, print_result=True):
    graph = Graph.build_graph(config.n_stages, config.n_micro, config)
    best_time, order, complete_time = initial_solution(graph, print_result)
    return create_scheduled_nodes(graph, complete_time)


def create_scheduled_nodes(graph, completion_time):
    typenames = [FuncType.F, FuncType.B, FuncType.W]
    cats = {
        FuncType.F: 0,
        FuncType.B: 1,
        FuncType.W: 2,
    }
    local_order = []
    end_time = []
    for t in completion_time:
        end_time.append(t)
    for stage in range(graph.nstages):
        order = []
        for cat in range(3):
            for mb in range(graph.nmb):
                order.append(
                    ScheduledNode(
                        type=typenames[cat],
                        stage=stage,
                        microbatch=mb,
                        layer_group_idx=stage,
                    )
                )
        order = sorted(order, key=lambda n: completion_time[graph.get_id(cats[n.type], n.stage, n.microbatch)])
        local_order.append(order)
    return local_order


def add_point_to_point_nodes(compute_order):
    schedule = []
    for node in compute_order:
        if node.type == FuncType.F:
            schedule.extend(
                [
                    ScheduledNode(FuncType.RECV_FORWARD, node.microbatch, node.stage),
                    node,
                    ScheduledNode(FuncType.SEND_FORWARD, node.microbatch, node.stage),
                ]
            )
        elif node.type == FuncType.B:
            schedule.extend(
                [
                    ScheduledNode(FuncType.RECV_BACKWARD, node.microbatch, node.stage),
                    node,
                    ScheduledNode(FuncType.SEND_BACKWARD, node.microbatch, node.stage),
                ]
            )
        else:
            schedule.append(node)
    return schedule


_SCHEDULE_CACHE = {}


def _get_cache_key(config: GraphConfig):
    return (
        tuple(config.cost_f),
        tuple(config.cost_b),
        tuple(config.cost_w),
        config.cost_comm,
        tuple(config.mem_f),
        tuple(config.mem_b),
        tuple(config.mem_w),
        tuple(config.max_mem) if config.max_mem is not None else None,
        config.print_scaling,
        config.max_chunks,
        config.n_stages,
        config.n_micro,
    )


def get_zb_schedule(num_microbatches: int, pipeline_parallel_size: int, pipeline_parallel_rank: int):
    config = GraphConfig(
        cost_f=[1.0] * pipeline_parallel_size,
        cost_b=[1.0] * pipeline_parallel_size,
        cost_w=[1.0] * pipeline_parallel_size,
        cost_comm=0.0,
        mem_f=[1.0] * pipeline_parallel_size,
        mem_b=[-1.0] * pipeline_parallel_size,
        mem_w=[0.0] * pipeline_parallel_size,
        max_mem=None,
        n_stages=pipeline_parallel_size,
        n_micro=num_microbatches,
    )
    cache_key = _get_cache_key(config)
    if cache_key not in _SCHEDULE_CACHE:
        _SCHEDULE_CACHE[cache_key] = [
            add_point_to_point_nodes(stage_order) for stage_order in create_schedule(config, print_result=False)
        ]
    return _SCHEDULE_CACHE[cache_key][pipeline_parallel_rank]
