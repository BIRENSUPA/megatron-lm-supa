from .communication import validate_communication


def run_schedule_passes(schedule):
    validate_communication(schedule)
    return schedule
