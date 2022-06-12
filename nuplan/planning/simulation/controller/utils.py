def forward_integrate(init: float, delta: float, sampling_time: float) -> float:
    """
    Performs a simple euler integration
    :param init: initial state
    :param delta: the rate of chance of the state
    :return: The result of integration
    """
    return init + delta * sampling_time
