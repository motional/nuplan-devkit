import unittest

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder, _create_scenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import Detections
from nuplan.planning.simulation.planner.ml_planner.ml_planner import MLPlanner
from nuplan.planning.simulation.simulation_manager.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.models.raster_model import RasterModel
from nuplan.planning.training.modeling.models.simple_vector_map_model import VectorMapSimpleMLP
from nuplan.planning.training.preprocessing.feature_builders.raster_feature_builder import RasterFeatureBuilder
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import \
    EgoTrajectoryTargetBuilder


def construct_simple_vector_map_ml_planner() -> MLPlanner:
    """
    Construct vector map simple planner
    :return: MLPlanner with vector map model
    """
    # Create model
    future_trajectory_param = TrajectorySampling(time_horizon=6.0, num_poses=12)
    past_trajectory_param = TrajectorySampling(time_horizon=2.0, num_poses=4)

    model = VectorMapSimpleMLP(num_output_features=36,
                               hidden_size=128,
                               vector_map_feature_radius=20,
                               past_trajectory_sampling=past_trajectory_param,
                               future_trajectory_sampling=future_trajectory_param)
    # Create planner
    return MLPlanner(model=model)


def construct_raster_ml_planner() -> MLPlanner:
    """
    Construct Raster ML Planner
    :return: MLPlanner with raster model
    """
    # Create model
    future_trajectory_param = TrajectorySampling(time_horizon=6.0, num_poses=12)

    model = RasterModel(
        model_name="resnet50",
        pretrained=True,
        num_input_channels=4,
        num_features_per_pose=3,
        future_trajectory_sampling=future_trajectory_param,
        feature_builders=[
            RasterFeatureBuilder(
                map_features={
                    'LANE': 255, 'INTERSECTION': 255,
                    'STOP_LINE': 128, 'CROSSWALK': 128},
                num_input_channels=4,
                target_width=224,
                target_height=224,
                target_pixel_size=0.5,
                ego_width=2.297,
                ego_front_length=4.049,
                ego_rear_length=1.127,
                ego_longitudinal_offset=0.0,
                baseline_path_thickness=1,
            )],
        target_builders=[EgoTrajectoryTargetBuilder(
            future_trajectory_sampling=future_trajectory_param
        )])
    # Create planner
    return MLPlanner(model=model)


class TestMlPlanner(unittest.TestCase):
    """
    Test MLPlanner with two models
    """

    def setUp(self) -> None:
        # Extract a scenario
        self.scenario_builder = NuPlanScenarioBuilder(version="nuplan_v0.1_mini", data_root="/data/sets/nuplan")
        self.scenario = _create_scenario(db=self.scenario_builder._db,
                                         scenario_tuple=("unknown", self.scenario_builder._db.lidar_pc[10000].token),
                                         scenario_mapping=ScenarioMapping({}),
                                         vehicle_parameters=self.scenario_builder._vehicle_parameters)

    def test_simple_vector_net_model(self) -> None:
        """ Test Model Vector Map Simple """
        # Create planner
        self.run_test_ml_planner(construct_simple_vector_map_ml_planner())

    def test_raster_net_model(self) -> None:
        """ Test Raster Net model """
        # Create planner
        self.run_test_ml_planner(construct_raster_ml_planner())

    def run_test_ml_planner(self, planner: MLPlanner) -> None:
        """ Tests if progress is calculated correctly """
        scenario = self.scenario

        # Initialize History
        simulation_history_buffer_duration = 2
        buffer_size = int(simulation_history_buffer_duration / self.scenario.database_interval + 1)
        history = SimulationHistoryBuffer(buffer_size=buffer_size)
        history.initialize(scenario, Detections)

        # Initialize Planner
        planner.initialize(expert_goal_state=scenario.get_expert_goal_state(),
                           mission_goal=scenario.get_mission_goal(),
                           map_name=scenario.map_api.map_name,
                           map_api=scenario.map_api)
        # Compute Trajectory
        trajectory = planner.compute_trajectory(
            iteration=SimulationIteration(index=0, time_point=scenario.start_time), history=history)
        self.assertNotEqual(trajectory, None)
        # +1 because the predicted trajectory does not contain ego's initial state
        self.assertEqual(len(trajectory.get_sampled_trajectory()), planner._num_output_dim + 1)


if __name__ == '__main__':
    unittest.main()
