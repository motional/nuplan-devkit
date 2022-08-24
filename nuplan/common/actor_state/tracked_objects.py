from __future__ import annotations

from functools import cached_property
from typing import Dict, Iterable, List, Optional, Tuple, Union

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.agent_temporal_state import AgentTemporalState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObject, SceneObjectMetadata
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, STATIC_OBJECT_TYPES, TrackedObjectType

TrackedObject = Union[Agent, StaticObject, SceneObject, AgentTemporalState]


class TrackedObjects:
    """Class representing tracked objects, a collection of SceneObjects"""

    def __init__(self, tracked_objects: Optional[List[TrackedObject]] = None):
        """
        :param tracked_objects: List of tracked objects
        """
        tracked_objects = tracked_objects if tracked_objects is not None else []

        self.tracked_objects = sorted(
            tracked_objects, key=lambda agent: agent.tracked_object_type.value  # type: ignore
        )

    def __iter__(self) -> Iterable[TrackedObject]:
        """When iterating return the tracked objects."""
        return iter(self.tracked_objects)

    @classmethod
    def from_oriented_boxes(cls, boxes: List[OrientedBox]) -> TrackedObjects:
        """When iterating return the tracked objects."""
        scene_objects = [
            SceneObject(
                TrackedObjectType.GENERIC_OBJECT,
                box,
                SceneObjectMetadata(timestamp_us=i, token=str(i), track_token=None, track_id=None),
            )
            for i, box in enumerate(boxes)
        ]
        return TrackedObjects(scene_objects)

    @cached_property
    def _ranges_per_type(self) -> Dict[TrackedObjectType, Tuple[int, int]]:
        """
        Returns the start and end index of the range of agents for each agent type
        in the list of agents (sorted by agent type). The ranges are cached for subsequent calls.
        """
        ranges_per_type: Dict[TrackedObjectType, Tuple[int, int]] = {}

        if self.tracked_objects:
            last_agent_type = self.tracked_objects[0].tracked_object_type
            start_range = 0
            end_range = len(self.tracked_objects)

            for idx, agent in enumerate(self.tracked_objects):
                if agent.tracked_object_type is not last_agent_type:
                    ranges_per_type[last_agent_type] = (start_range, idx)
                    start_range = idx
                    last_agent_type = agent.tracked_object_type
            ranges_per_type[last_agent_type] = (start_range, end_range)

            ranges_per_type.update(
                {
                    agent_type: (end_range, end_range)
                    for agent_type in TrackedObjectType
                    if agent_type not in ranges_per_type
                }
            )

        return ranges_per_type

    def get_tracked_objects_of_type(self, tracked_object_type: TrackedObjectType) -> List[TrackedObject]:
        """
        Gets the sublist of agents of a particular TrackedObjectType
        :param tracked_object_type: The query TrackedObjectType
        :return: List of the present agents of the query type. Throws an error if the key is invalid.
        """
        if tracked_object_type in self._ranges_per_type:
            start_idx, end_idx = self._ranges_per_type[tracked_object_type]
            return self.tracked_objects[start_idx:end_idx]

        else:
            # There are no objects of the queried type
            return []

    def get_agents(self) -> List[Agent]:
        """
        Getter for the tracked objects which are Agents
        :return: list of Agents
        """
        agents = []
        for agent_type in AGENT_TYPES:
            agents.extend(self.get_tracked_objects_of_type(agent_type))
        return agents

    def get_static_objects(self) -> List[StaticObject]:
        """
        Getter for the tracked objects which are StaticObjects
        :return: list of StaticObjects
        """
        static_objects = []
        for static_object_type in STATIC_OBJECT_TYPES:
            static_objects.extend(self.get_tracked_objects_of_type(static_object_type))
        return static_objects

    def __len__(self) -> int:
        """
        :return: The number of tracked objects in the class
        """
        return len(self.tracked_objects)

    def get_tracked_objects_of_types(self, tracked_object_types: List[TrackedObjectType]) -> List[TrackedObject]:
        """
        Gets the sublist of agents of particular TrackedObjectTypes
        :param tracked_object_types: The query TrackedObjectTypes
        :return: List of the present agents of the query types. Throws an error if the key is invalid.
        """
        open_loop_tracked_objects = []
        for _type in tracked_object_types:
            open_loop_tracked_objects.extend(self.get_tracked_objects_of_type(_type))

        return open_loop_tracked_objects
