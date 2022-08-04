from typing import Any, Dict, List

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.callback.serialization_callback import convert_sample_to_scene
from nuplan.planning.simulation.history.simulation_history import SimulationHistorySample
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.training.callbacks.utils.scene_converter import SceneConverter
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType


class ScenarioSceneConverter(SceneConverter):
    """
    Scene writer that converts a scenario sample to scene.
    """

    def __init__(self, ego_trajectory_horizon: float, ego_trajectory_poses: int) -> None:
        """
        Initialize scene writer.
        :param ego_trajectory_horizon: the horizon to get ego's future trajectory.
        :param ego_trajectory_poses: number of poses for ego's future trajectory.
        """
        self._ego_trajectory_horizon = ego_trajectory_horizon
        self._ego_trajectory_poses = ego_trajectory_poses

    def __call__(
        self, scenario: AbstractScenario, features: FeaturesType, targets: TargetsType, predictions: FeaturesType
    ) -> List[Dict[str, Any]]:
        """Inherited, see superclass."""
        index = 0  # Use initial index of the scenario

        ego_trajectory = [scenario.get_ego_state_at_iteration(index)] + list(
            scenario.get_ego_future_trajectory(index, self._ego_trajectory_horizon, self._ego_trajectory_poses)
        )  # Ego trajectory including initial state

        sample = SimulationHistorySample(
            iteration=SimulationIteration(time_point=scenario.get_time_point(index), index=index),
            ego_state=scenario.get_ego_state_at_iteration(index),
            trajectory=InterpolatedTrajectory(ego_trajectory),
            observation=scenario.get_tracked_objects_at_iteration(index),
            traffic_light_status=scenario.get_traffic_light_status_at_iteration(index),
        )

        scene = convert_sample_to_scene(
            map_name=scenario.map_api.map_name,
            database_interval=scenario.database_interval,
            traffic_light_status=scenario.get_traffic_light_status_at_iteration(index),
            expert_trajectory=ego_trajectory,
            mission_goal=scenario.get_mission_goal(),
            data=sample,
        )

        return [scene]
