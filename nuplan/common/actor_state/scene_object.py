from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType


@dataclass(frozen=True)
class SceneObjectMetadata:
    """
    Metadata for every object
    """

    # Timestamp of this object in micro seconds
    timestamp_us: int
    # Unique token in a whole dataset
    token: str
    # Human understandable id of the object
    track_id: Optional[int]
    # Token of the object which is temporally consistent
    track_token: Optional[str]
    # Human readable category name
    category_name: Optional[str] = None

    @property
    def timestamp_s(self) -> float:
        """
        :return: timestamp in seconds
        """
        return self.timestamp_us * 1e-6


class SceneObject:
    """Class describing SceneObjects, i.e. objects present in a planning scene"""

    def __init__(
        self, tracked_object_type: TrackedObjectType, oriented_box: OrientedBox, metadata: SceneObjectMetadata
    ):
        """
        Representation of an Agent in the scene.
        :param tracked_object_type: Type of the current static object
        :param oriented_box: Geometrical representation of the static object
        :param metadata: High-level information about the object
        """
        self._metadata = metadata
        self.instance_token = None
        self._tracked_object_type = tracked_object_type

        self._box: OrientedBox = oriented_box

    @property
    def metadata(self) -> SceneObjectMetadata:
        """
        Getter for object metadata
        :return: Object's metadata
        """
        return self._metadata

    @property
    def token(self) -> str:
        """
        Getter for object unique token, different for same object in different samples
        :return: The unique token
        """
        return self._metadata.token

    @property
    def track_token(self) -> Optional[str]:
        """
        Getter for object unique token tracked across samples, same for same objects in different samples
        :return: The unique track token
        """
        return self._metadata.track_token

    @property
    def tracked_object_type(self) -> TrackedObjectType:
        """
        Getter for object classification type
        :return: The object classification type
        """
        return self._tracked_object_type

    @property
    def box(self) -> OrientedBox:
        """
        Getter for object OrientedBox
        :return: The object oriented box
        """
        return self._box

    @property
    def center(self) -> StateSE2:
        """
        Getter for object center pose
        :return: The center pose
        """
        return self.box.center

    @classmethod
    def make_random(cls, token: str, object_type: TrackedObjectType) -> SceneObject:
        """
        Instantiates a random SceneObject.
        :param token: Unique token
        :param object_type: Classification type
        :return: SceneObject instance.
        """
        center = random.sample(range(50), 2)
        heading = np.random.uniform(-np.pi, np.pi)
        size = random.sample(range(1, 50), 3)
        track_id = random.sample(range(1, 10), 1)[0]
        timestamp_us = random.sample(range(1, 10), 1)[0]

        return SceneObject(
            metadata=SceneObjectMetadata(token=token, track_id=track_id, track_token=token, timestamp_us=timestamp_us),
            tracked_object_type=object_type,
            oriented_box=OrientedBox(StateSE2(*center, heading), size[0], size[1], size[2]),
        )

    @classmethod
    def from_raw_params(
        cls,
        token: str,
        track_token: str,
        timestamp_us: int,
        track_id: int,
        center: StateSE2,
        size: Tuple[float, float, float],
    ) -> SceneObject:
        """
        Instantiates a generic SceneObject.
        :param token: The token of the object.
        :param track_token: The track token of the object.
        :param timestamp_us: [us] timestamp for the object.
        :param track_id: Human readable track id.
        :param center: Center pose.
        :param size: Size of the geometrical box (width, length, height).
        :return: SceneObject instance.
        """
        box = OrientedBox(center, width=size[0], length=size[1], height=size[2])
        return SceneObject(
            metadata=SceneObjectMetadata(
                token=token, track_token=track_token, timestamp_us=timestamp_us, track_id=track_id
            ),
            tracked_object_type=TrackedObjectType.GENERIC_OBJECT,
            oriented_box=box,
        )
