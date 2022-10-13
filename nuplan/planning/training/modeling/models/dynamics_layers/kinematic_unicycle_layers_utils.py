from enum import IntEnum


class StateIndex(IntEnum):
    """
    Index mapping for the state vector
    """

    X_POS = 0  # [m] The x position in global coordinates.
    Y_POS = 1  # [m] The y position in global coordinates.
    YAW = 2  # [rad] The yaw in global coordinates.
    X_VELOCITY = 3  # [m/s] The velocity along the longitudinal axis of the vehicle.
    Y_VELOCITY = 4  # [m/s] The velocity along the lateral axis of the vehicle.
    X_ACCEL = 5  # [m/s^2] The acceleration along the longitudinal axis of the vehicle.
    Y_ACCEL = 6  # [m/s^2] The acceleration along the lateral axis of the vehicle.


class InputIndex(IntEnum):
    """
    Index mapping for the input vector
    """

    CURVATURE = 0  # [rad/m] The curvature.
    JERK = 1  # [m/s^3] The jerk along the longitudinal axis of the vehicle.
