import unittest

from nuplan.planning.simulation.observation.idm.idm_policy import IDMPolicy
from nuplan.planning.simulation.observation.idm.idm_states import IDMAgentState, IDMLeadAgentState


class IDMPolicyTests(unittest.TestCase):
    """Tests implementation of IDMPolicy"""

    def setUp(self):  # type: ignore
        """Test setup"""
        self.idm = IDMPolicy(
            target_velocity=30, min_gap_to_lead_agent=2, headway_time=1.5, accel_max=0.73, decel_max=1.67
        )
        self.sampling_time = 0.5
        self.agent = IDMAgentState(5, 3)
        self.lead_agent = IDMLeadAgentState(15, 2, 5)

    def test_idm_model(self):  # type: ignore
        """Tests the model correctness"""
        model = self.idm.idm_model([], self.agent.to_array(), self.lead_agent.to_array(), self.idm.idm_params)

        self.assertEqual(3, model[0])
        self.assertAlmostEqual(-1.073366, model[1])

    def test_solve_forward_euler_idm_policy(self):  # type: ignore
        """Tests expected behaviour of forward euler method"""
        solution = self.idm.solve_forward_euler_idm_policy(self.agent, self.lead_agent, self.sampling_time)
        self.assertEqual(6.5, solution.progress)
        self.assertAlmostEqual(2.46331699693, solution.velocity)

    def test_non_differential_idm_policy(self):  # type: ignore
        """Tests expected behaviour of odeint integrator"""
        solution = self.idm.solve_odeint_idm_policy(self.agent, self.lead_agent, self.sampling_time, 2)
        self.assertAlmostEqual(6.3558523392415, solution.progress)
        self.assertAlmostEqual(2.4058965769308, solution.velocity)

    def test_solve_ivp_idm_policy(self):  # type: ignore
        """Tests expected behaviour of inital value problem integrator"""
        solution = self.idm.solve_ivp_idm_policy(self.agent, self.lead_agent, self.sampling_time)
        self.assertAlmostEqual(6.355856711603, solution.progress)
        self.assertAlmostEqual(2.40590847399835, solution.velocity)


if __name__ == '__main__':
    unittest.main()
