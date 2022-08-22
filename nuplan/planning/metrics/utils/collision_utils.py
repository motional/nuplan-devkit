from enum import IntEnum
from typing import Dict, List

import numpy as np

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.scene_object import SceneObject
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.metrics.metric_result import MetricStatisticsType, Statistic

VRU_types = [
    TrackedObjectType.PEDESTRIAN,
    TrackedObjectType.BICYCLE,
]

object_types = [
    TrackedObjectType.TRAFFIC_CONE,
    TrackedObjectType.BARRIER,
    TrackedObjectType.CZONE_SIGN,
    TrackedObjectType.GENERIC_OBJECT,
]


class CollisionType(IntEnum):
    """Enum for the types of collisions of interest."""

    STOPPED_EGO_COLLISION = 0
    STOPPED_TRACK_COLLISION = 1
    ACTIVE_FRONT_COLLISION = 2
    ACTIVE_REAR_COLLISION = 3
    ACTIVE_LATERAL_COLLISION = 4


def ego_delta_v_collision(
    ego_state: EgoState, scene_object: SceneObject, ego_mass: float = 2000, agent_mass: float = 2000
) -> float:
    """
    Compute the ego delta V (loss of velocity during the collision). Delta V represents the intensity of the collision
    of the ego with other agents.
    :param ego_state: The state of ego.
    :param scene_object: The scene_object ego is colliding with.
    :param ego_mass: mass of ego.
    :param agent_mass: mass of the agent.
    :return The delta V measure for ego.
    """
    # Collision metric is defined as the ratio of agent mass to the overall mass of ego and agent, times the changes in velocity defined as
    # sqrt(ego_speed^2 + agent_speed^2 - 2 * ego_speed * agent_speed * cos(heading_difference))
    ego_mass_ratio = agent_mass / (agent_mass + ego_mass)

    scene_object_speed = scene_object.velocity.magnitude() if isinstance(scene_object, Agent) else 0

    sum_speed_squared = ego_state.dynamic_car_state.speed**2 + scene_object_speed**2
    cos_rule_term = (
        2
        * ego_state.dynamic_car_state.speed
        * scene_object_speed
        * np.cos(ego_state.rear_axle.heading - scene_object.center.heading)
    )
    velocity_component = float(np.sqrt(sum_speed_squared - cos_rule_term))

    return ego_mass_ratio * velocity_component


def get_fault_type_statistics(
    all_at_fault_collisions: Dict[TrackedObjectType, List[float]],
) -> List[Statistic]:
    """
    :param all_at_fault_collisions: Dict of at_fault collisions.
    :return: List of Statistics for all collision track types.
    """
    statistics = []
    track_types_collisions_energy_dict: Dict[str, List[float]] = {}

    for collision_track_type, collision_name in zip(
        [VRU_types, [TrackedObjectType.VEHICLE], object_types], ['VRUs', 'vehicles', 'objects']
    ):
        track_types_collisions_energy_dict[collision_name] = [
            colision_energy
            for track_type in collision_track_type
            for colision_energy in all_at_fault_collisions[track_type]
        ]
        statistics.extend(
            [
                Statistic(
                    name=f'number_of_at_fault_collisions_with_{collision_name}',
                    unit=MetricStatisticsType.COUNT.unit,
                    value=len(track_types_collisions_energy_dict[collision_name]),
                    type=MetricStatisticsType.COUNT,
                )
            ]
        )
    for collision_name, track_types_collisions_energy in track_types_collisions_energy_dict.items():
        if len(track_types_collisions_energy) > 0:
            statistics.extend(
                [
                    Statistic(
                        name=f'max_collision_energy_with_{collision_name}',
                        unit="meters_per_second",
                        value=max(track_types_collisions_energy),
                        type=MetricStatisticsType.MAX,
                    ),
                    Statistic(
                        name=f'min_collision_energy_with_{collision_name}',
                        unit="meters_per_second",
                        value=min(track_types_collisions_energy),
                        type=MetricStatisticsType.MIN,
                    ),
                    Statistic(
                        name=f'mean_collision_energy_with_{collision_name}',
                        unit="meters_per_second",
                        value=np.mean(track_types_collisions_energy),
                        type=MetricStatisticsType.MEAN,
                    ),
                ]
            )
    return statistics
