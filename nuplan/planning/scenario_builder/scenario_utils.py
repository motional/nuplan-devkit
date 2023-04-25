from typing import List


def sample_indices_with_time_horizon(num_samples: int, time_horizon: float, time_interval: float) -> List[int]:
    """
    Samples the indices that can access N number of samples in a T time horizon from a sequence
    of temporal elemements with DT time interval.
    :param num_samples: number of elements to sample.
    :param time_horizon: [s] time horizon of sampled elements.
    :param time_interval: [s] time interval of sequence to sample from.
    :return: sampled indices that access the temporal sequence.
    """
    if time_horizon <= 0.0 or time_interval <= 0.0 or time_horizon < time_interval:
        raise ValueError(
            f'Time horizon {time_horizon} must be greater or equal than target time interval {time_interval}'
            ' and both must be positive.'
        )

    # Compute step size and number of intervals to sample from
    num_intervals = int(time_horizon / time_interval) + 1
    step_size = num_intervals // num_samples

    assert step_size > 0, f"Cannot get {num_samples} samples in a {time_horizon}s horizon at {time_interval}s intervals"

    # Compute the indices
    indices = list(range(step_size, num_intervals + 1, step_size))
    indices = indices[:num_samples]

    assert len(indices) == num_samples, f'Expected {num_samples} samples but only {len(indices)} were sampled'

    return indices
