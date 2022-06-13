import abc

from nuplan.common.actor_state.ego_state import EgoState, EgoStateDot


class AbstractMotionModel(abc.ABC):
    """
    Interface for generic eo controllers.
    """

    @abc.abstractmethod
    def get_state_dot(self, state: EgoState) -> EgoStateDot:
        """
        Compute x_dot = f(x) for the motion model.

        :param state: The state for which to compute motion model.
        :return: The state derivative as an EgoState.
        """
        pass

    @abc.abstractmethod
    def propagate_state(self, state: EgoState, sampling_time: float) -> EgoState:
        """
        Propagate the state according to the motion model.

        :param state: The initial state to propagate.
        :param sampling_time: [s] The time duration to propagate for.
        :return: The propagated state.
        """
        pass
