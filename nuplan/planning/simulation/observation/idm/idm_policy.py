from math import sqrt
from typing import Any, List

import numpy as np
from scipy.integrate import odeint, solve_ivp

from nuplan.planning.simulation.observation.idm.idm_states import IDMAgentState, IDMLeadAgentState


class IDMPolicy:
    """
    An agent policy that describes the agent's behaviour w.r.t to a lead agent. The policy only controls the
    longitudinal states (progress, velocity) of the agent. This longitudinal states are used to propagate the agent
    along a given path.
    """

    def __init__(
        self,
        target_velocity: float,
        min_gap_to_lead_agent: float,
        headway_time: float,
        accel_max: float,
        decel_max: float,
    ):
        """
        Constructor for IDMPolicy

        :param target_velocity: Desired velocity in free traffic [m/s]
        :param min_gap_to_lead_agent: Minimum relative distance to lead vehicle [m]
        :param headway_time: Desired time headway. The minimum possible time to the vehicle in front [s]
        :param accel_max: maximum acceleration [m/s^2]
        :param decel_max: maximum deceleration (positive value) [m/s^2]
        """
        self._target_velocity = target_velocity
        self._min_gap_to_lead_agent = min_gap_to_lead_agent
        self._headway_time = headway_time
        self._accel_max = accel_max
        self._decel_max = decel_max

    @property
    def idm_params(self) -> List[float]:
        """Returns the policy parameters as a list"""
        return [
            self._target_velocity,
            self._min_gap_to_lead_agent,
            self._headway_time,
            self._accel_max,
            self._decel_max,
        ]

    @property
    def target_velocity(self) -> float:
        """
        The policy's desired velocity in free traffic [m/s]
        :return: target velocity
        """
        return self._target_velocity

    @target_velocity.setter
    def target_velocity(self, target_velocity: float) -> None:
        """
        Sets the policy's desired velocity in free traffic [m/s]
        """
        self._target_velocity = target_velocity
        assert target_velocity > 0, f"The target velocity must be greater than 0! {target_velocity} > 0"

    @property
    def headway_time(self) -> float:
        """
        The policy's minimum possible time to the vehicle in front [s]
        :return: Desired time headway
        """
        return self._headway_time

    @property
    def decel_max(self) -> float:
        """
        The policy's maximum deceleration (positive value) [m/s^2]
        :return: Maximum deceleration
        """
        return self._decel_max

    @staticmethod
    def idm_model(
        time_points: List[float], state_variables: List[float], lead_agent: List[float], params: List[float]
    ) -> List[Any]:
        """
        Defines the differential equations for IDM.

        :param state_variables: vector of the state variables:
                  state_variables = [x_agent: progress,
                                     v_agent: velocity]
        :param time_points: time A sequence of time points for which to solve for the state variables
        :param lead_agent: vector of the state variables for the lead vehicle:
                  lead_agent = [x_lead: progress,
                                v_lead: velocity,
                                l_r_lead: half length of the leading vehicle]
        :param params:vector of the parameters:
                  params = [target_velocity: desired velocity in free traffic,
                            min_gap_to_lead_agent: minimum relative distance to lead vehicle,
                            headway_time: desired time headway. The minimum possible time to the vehicle in front,
                            accel_max: maximum acceleration,
                            decel_max: maximum deceleration (positive value)]

        :return: system of differential equations
        """
        # state variables
        x_agent, v_agent = state_variables
        x_lead, v_lead, l_r_lead = lead_agent

        # parameters
        target_velocity, min_gap_to_lead_agent, headway_time, accel_max, decel_max = params
        acceleration_exponent = 4  # Usually set to 4

        # convenience definitions
        s_star = (
            min_gap_to_lead_agent
            + v_agent * headway_time
            + (v_agent * (v_agent - v_lead)) / (2 * sqrt(accel_max * decel_max))
        )
        s_alpha = max(x_lead - x_agent - l_r_lead, min_gap_to_lead_agent)  # clamp to avoid zero division

        # differential equations
        x_dot = v_agent
        v_agent_dot = accel_max * (1 - (v_agent / target_velocity) ** acceleration_exponent - (s_star / s_alpha) ** 2)

        return [x_dot, v_agent_dot]

    def solve_forward_euler_idm_policy(
        self, agent: IDMAgentState, lead_agent: IDMLeadAgentState, sampling_time: float
    ) -> IDMAgentState:
        """
        Solves Solves an initial value problem for a system of ODEs using forward euler.
        This has the benefit of being differentiable

        :param agent: the agent of interest
        :param lead_agent: the lead vehicle
        :param sampling_time: interval of integration
        :return: solution to the differential equations
        """
        params = self.idm_params

        x_dot, v_agent_dot = self.idm_model([], agent.to_array(), lead_agent.to_array(), params)

        return IDMAgentState(
            agent.progress + sampling_time * x_dot,
            agent.velocity + sampling_time * min(max(-self._decel_max, v_agent_dot), self._accel_max),
        )

    def solve_odeint_idm_policy(
        self, agent: IDMAgentState, lead_agent: IDMLeadAgentState, sampling_time: float, solve_points: int = 10
    ) -> IDMAgentState:
        """
        Solves an initial value problem for a system of ODEs using scipy odeint

        :param agent: the agent of interest
        :param lead_agent: the lead vehicle
        :param sampling_time: interval of integration
        :param solve_points: number of points for temporal resolution
        :return: solution to the differential equations
        """
        t = np.linspace(0, sampling_time, solve_points)
        solution = odeint(
            self.idm_model,
            agent.to_array(),
            t,
            args=(
                lead_agent.to_array(),
                self.idm_params,
            ),
            tfirst=True,
        )

        # return the last solution
        return IDMAgentState(solution[-1][0], solution[-1][1])

    def solve_ivp_idm_policy(
        self, agent: IDMAgentState, lead_agent: IDMLeadAgentState, sampling_time: float
    ) -> IDMAgentState:
        """
        Solves an initial value problem for a system of ODEs using scipy RK45

        :param agent: the agent of interest
        :param lead_agent: the lead vehicle
        :param sampling_time: interval of integration
        :return: solution to the differential equations
        """
        t = (0, sampling_time)
        solution = solve_ivp(
            self.idm_model,
            t,
            agent.to_array(),
            args=(
                lead_agent.to_array(),
                self.idm_params,
            ),
            method='RK45',
        )

        # return the last solution
        return IDMAgentState(solution.y[0][-1], solution.y[1][-1])
