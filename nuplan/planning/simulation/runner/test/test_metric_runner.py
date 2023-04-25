import tempfile
import unittest
from pathlib import Path

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.planning.metrics.metric_engine import MetricsEngine
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.callback.metric_callback import MetricCallback
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.simple_planner import SimplePlanner
from nuplan.planning.simulation.runner.metric_runner import MetricRunner
from nuplan.planning.simulation.simulation_log import SimulationLog
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


class TestMetricRunner(unittest.TestCase):
    """Tests MetricRunner class which is computing metric."""

    def setUp(self) -> None:
        """Setup Mock classes."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.scenario = MockAbstractScenario(number_of_past_iterations=10)
        self.history = SimulationHistory(self.scenario.map_api, self.scenario.get_mission_goal())
        state_0 = EgoState.build_from_rear_axle(
            StateSE2(0, 0, 0),
            vehicle_parameters=self.scenario.ego_vehicle_parameters,
            rear_axle_velocity_2d=StateVector2D(x=0, y=0),
            rear_axle_acceleration_2d=StateVector2D(x=0, y=0),
            tire_steering_angle=0,
            time_point=TimePoint(0),
        )
        state_1 = EgoState.build_from_rear_axle(
            StateSE2(0, 0, 0),
            vehicle_parameters=self.scenario.ego_vehicle_parameters,
            rear_axle_velocity_2d=StateVector2D(x=0, y=0),
            rear_axle_acceleration_2d=StateVector2D(x=0, y=0),
            tire_steering_angle=0,
            time_point=TimePoint(1000),
        )
        self.history.add_sample(
            SimulationHistorySample(
                iteration=SimulationIteration(time_point=TimePoint(0), index=0),
                ego_state=state_0,
                trajectory=InterpolatedTrajectory(trajectory=[state_0, state_1]),
                observation=DetectionsTracks(TrackedObjects()),
                traffic_light_status=self.scenario.get_traffic_light_status_at_iteration(0),
            )
        )
        self.history.add_sample(
            SimulationHistorySample(
                iteration=SimulationIteration(time_point=TimePoint(0), index=0),
                ego_state=state_1,
                trajectory=InterpolatedTrajectory(trajectory=[state_0, state_1]),
                observation=DetectionsTracks(TrackedObjects()),
                traffic_light_status=self.scenario.get_traffic_light_status_at_iteration(0),
            )
        )

        save_path = Path(self.tmp_dir.name)
        planner = SimplePlanner(2, 0.5, [0, 0])

        self.simulation_log = SimulationLog(
            file_path=save_path / 'simulation_logs',
            simulation_history=self.history,
            scenario=self.scenario,
            planner=planner,
        )

        self.metric_engine = MetricsEngine(metrics=[], main_save_path=save_path / 'metrics')
        self.metric_callback = MetricCallback(metric_engine=self.metric_engine)
        self.metric_runner = MetricRunner(simulation_log=self.simulation_log, metric_callback=self.metric_callback)

    def tearDown(self) -> None:
        """Clean up folders."""
        self.tmp_dir.cleanup()

    def test_run_metric_runner(self) -> None:
        """Test to run metric_runner."""
        self.metric_runner.run()


if __name__ == '__main__':
    unittest.main()
