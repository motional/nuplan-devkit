from nuplan.common.actor_state.state_representation import TimePoint


def forward_integrate(init: float, delta: float, sampling_time: TimePoint) -> float:
    """
    Performs a simple euler integration.
    :param init: Initial state
    :param delta: The rate of chance of the state.
    :param sampling_time: The time duration to propagate for.
    :return: The result of integration
    """
    return float(init + delta * sampling_time.time_s)
