from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List

import numpy as np

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.scene_object import SceneObject
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.metrics.metric_result import MetricViolation


class CollisionType(IntEnum):
    """Enum for the types of collisions of interest."""

    STOPPED_EGO_COLLISION = 0
    STOPPED_TRACK_COLLISION = 1
    ACTIVE_FRONT_COLLISION = 2
    ACTIVE_REAR_COLLISION = 3
    ACTIVE_LATERAL_COLLISION = 4


@dataclass
class AtFaultCollision:
    """
    Class to retain information about the collisions at a particular timestamp.
    """

    timestamp: int  # The timestamp at time of collision
    duration: float  # Duration of collision, we assume collision is only evaluated at the first timestamp of collision
    collision_ego_delta_v: float  # Energy in collision
    collision_type: CollisionType  # Type of collision


def ego_delta_v_collision(
    ego_state: EgoState, scene_object: SceneObject, ego_mass: float = 2000, agent_mass: float = 2000
) -> float:
    """
    Computes the ego delta V (loss of velocity during the collision). Delta V represents the intensity of the collision
    of the ego with other agents.
    :param ego_state: The state of ego
    :param scene_object: The scene_object ego is colliding with
    :param ego_mass: mass of ego
    :param agent_mass: mass of the agent
    :return The delta V measure for ego
    """
    # Collision metric is defined as the ratio of agent mass to the overall mass of ego and agent, times the changes in velocity defined as sqrt(ego_speed^2 + agent_speed^2 - 2 * ego_speed * agent_speed * cos(heading_difference))
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


def compute_collision_violation(
    at_fault_collision: AtFaultCollision,
    metric_name: str,
    metric_category: str,
) -> MetricViolation:
    """
    Computes the violation metric for an at fault collision.

    :param metric_name: Metric name
    :param metric_category: Metric category
    :return: MetricViolation at the current timestamp.
    """
    return MetricViolation(
        metric_computator=metric_name,
        name=metric_name,
        metric_category=metric_category,
        unit="meters_per_second",
        start_timestamp=at_fault_collision.timestamp,
        duration=at_fault_collision.duration,
        extremum=at_fault_collision.collision_ego_delta_v,
        mean=at_fault_collision.collision_ego_delta_v,
    )


def get_fault_type_violation(
    track_types: List[TrackedObjectType],
    all_at_fault_collisions: Dict[TrackedObjectType, List[AtFaultCollision]],
    name: str,
    category: str,
) -> List[MetricViolation]:
    """
    :param object_types: List of track types
    :param all_at_fault_collisions: Dict of at_fault collisions
    :param name: metric name
    :param category: metric category
    :return: List of MetricViolation for the given track type
    """
    all_violations = [
        compute_collision_violation(
            at_fault_collision,
            name,
            category,
        )
        for track_type in track_types
        for at_fault_collision in all_at_fault_collisions[track_type]
    ]
    return all_violations
