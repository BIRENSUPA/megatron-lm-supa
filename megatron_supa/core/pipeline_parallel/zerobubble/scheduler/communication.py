from .graph import FuncType

_SEND_RECV_PAIRS = {
    FuncType.SEND_FORWARD: FuncType.RECV_FORWARD,
    FuncType.SEND_BACKWARD: FuncType.RECV_BACKWARD,
}


def validate_communication(schedule):
    counts = {func_type: 0 for func_type in FuncType}
    for node in schedule:
        counts[node.type] += 1
    for send_type, recv_type in _SEND_RECV_PAIRS.items():
        send_count = counts[send_type]
        recv_count = counts[recv_type]
        if send_count != recv_count and send_count != 0 and recv_count != 0:
            raise RuntimeError(
                f'Unbalanced local ZB communication nodes: {send_type.name}={send_count}, '
                f'{recv_type.name}={recv_count}'
            )
