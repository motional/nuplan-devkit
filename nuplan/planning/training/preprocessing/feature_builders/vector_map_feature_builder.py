from __future__ import annotations

from typing import Dict, List, Optional, Set, Type, cast

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.maps.maps_datatypes import LaneSegmentConnections, LaneSegmentCoords
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import Observation
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder, \
    FeatureBuilderMetaData
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap


def _transform_to_relative_frame(coords: npt.NDArray[np.float32],
                                 anchor_state: StateSE2) -> npt.NDArray[np.float32]:
    """
    Transform a set of coordinates to the the given frame.
    :param coords: <np.ndarray: num_coords, 2> Coordinates to be transformed.
    :param anchor_state: The coordinate frame to transform to.
    :return: <np.ndarray: num_coords, 2> Transformed coordinates.
    """
    # Extract transform
    transform = np.linalg.inv(anchor_state.as_matrix())  # type: ignore
    # Homogenous Coordinates
    coords = np.pad(coords, ((0, 0), (0, 1)), 'constant', constant_values=1.0)  # type: ignore
    coords = transform @ coords.transpose()
    return cast(npt.NDArray[np.float32], coords.transpose()[:, :2])


def _accumulate_connections(node_idx_to_neighbor_dict: Dict[int, Dict[str, Set[int]]], scales: List[int]) \
        -> Dict[int, npt.NDArray[np.float32]]:
    """
    Accumulate the connections over multiple scales
    :param node_idx_to_neighbor_dict: {node_idx: neighbor_dict} where each neighbor_dict
                                      will have format {'i_hop_neighbors': set_of_i_hop_neighbors}
    :param scales: Connections scales to generate.
    :return: Multi-scale connections as a dict of {scale: connections_of_scale}.
    """
    # Get the connections of each scale.
    multi_scale_connections = {}
    for scale in scales:
        scale_connections = []
        for node_idx, neighbor_dict in node_idx_to_neighbor_dict.items():
            for n_hop_neighbor in neighbor_dict[f"{scale}_hop_neighbors"]:
                scale_connections.append([node_idx, n_hop_neighbor])

        # if cannot find n-hop neighbors, return empty connection with size [0,2]
        if len(scale_connections) == 0:
            scale_connections = np.empty([0, 2], dtype=np.int64)  # type: ignore

        multi_scale_connections[scale] = np.array(scale_connections)

    return multi_scale_connections


def _generate_multi_scale_connections(connections: npt.NDArray[np.int32],
                                      scales: List[int]) -> Dict[int, npt.NDArray[np.float32]]:
    """
    Generate multi-scale connections by finding the neighbors up to max(scales) hops away for each node.
    :param connections: <np.ndarray: num_connections, 2>. A 1-hop connection is represented by [start_idx, end_idx]
    :param scales: Connections scales to generate.
    :return: Multi-scale connections as a dict of {scale: connections_of_scale}.
             Each connections_of_scale is represented by an array of <np.ndarray: num_connections, 2>,
    """
    # This dict will have format {node_idx: neighbor_dict},
    # where each neighbor_dict will have format {'i_hop_neighbors': set_of_i_hop_neighbors}.
    node_idx_to_neighbor_dict: Dict[int, Dict[str, Set[int]]] = {}

    # Initialize the data structure for each node with its 1-hop neighbors.
    for connection in connections:
        start_idx, end_idx = list(connection)
        if start_idx not in node_idx_to_neighbor_dict:
            node_idx_to_neighbor_dict[start_idx] = {"1_hop_neighbors": set()}
        if end_idx not in node_idx_to_neighbor_dict:
            node_idx_to_neighbor_dict[end_idx] = {"1_hop_neighbors": set()}
        node_idx_to_neighbor_dict[start_idx]["1_hop_neighbors"].add(end_idx)

    # Find the neighbors up to max(scales) hops away for each node.
    for scale in range(2, max(scales) + 1):
        for neighbor_dict in node_idx_to_neighbor_dict.values():
            neighbor_dict[f"{scale}_hop_neighbors"] = set()
            for n_hop_neighbor in neighbor_dict[f"{scale - 1}_hop_neighbors"]:
                for n_plus_1_hop_neighbor in node_idx_to_neighbor_dict[n_hop_neighbor]["1_hop_neighbors"]:
                    neighbor_dict[f"{scale}_hop_neighbors"].add(n_plus_1_hop_neighbor)

    return _accumulate_connections(node_idx_to_neighbor_dict, scales)


class VectorMapFeatureBuilder(AbstractFeatureBuilder):
    """
    Vector map feature builder. Note: this builder uses the feature builder that is used for LSN
    """

    def __init__(self, radius: float, connection_scales: Optional[List[int]] = None) -> None:
        """
        Initialize vector map builder with configuration parameters.
        :param radius:  The query radius scope relative to the current ego-pose.
        :param connection_scales: Connection scales to generate. Use the 1-hop connections if it's left empty.
        :return: Vector map data including lane segment coordinates and connections within the given range.
        """
        self._radius = radius
        self._connection_scales = connection_scales

    def get_feature_type(self) -> Type[AbstractModelFeature]:
        """ Inherited, see superclass. """
        return VectorMap  # type: ignore

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """ Inherited, see superclass. """
        return "vector_map"

    def get_features_from_scenario(self, scenario: AbstractScenario) -> VectorMap:
        """ Inherited, see superclass. """

        ego_state = scenario.initial_ego_state
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        ls_coords, ls_conns, ls_meta = scenario.map_api.get_neighbor_vector_map(ego_coords, self._radius)

        return self._compute_feature(ls_coords, ls_conns, ego_state.rear_axle)

    def get_features_from_simulation(self, ego_states: List[EgoState], observations: List[Observation],
                                     meta_data: FeatureBuilderMetaData) -> VectorMap:
        """ Inherited, see superclass. """

        ego_state = ego_states[-1]
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        ls_coords, ls_conns, ls_meta = meta_data.map_api.get_neighbor_vector_map(ego_coords, self._radius)

        return self._compute_feature(ls_coords, ls_conns, ego_state.rear_axle)

    def _compute_feature(self,
                         lane_coords: LaneSegmentCoords,
                         lane_conns: LaneSegmentConnections,
                         anchor_state: StateSE2) -> VectorMap:
        """
        :param lane_coords: A list of lane_segment coords in shape of [num_lane_segment, 2, 2]
        :param lane_conns: a List of lane_segment connections [start_idx, end_idx] in shape of [num_connection, 2]
        :return: VectorMap
        """

        lane_segment_coords = np.asarray(lane_coords.to_vector(), np.float32)
        lane_segment_conns = np.asarray(lane_conns.to_vector(), np.int64)

        # Transform the lane coordinates from global frame to ego vehicle frame.
        # Flatten lane_segment_coords from (num_lane_segment, 2, 2) to (num_lane_segment * 2, 2) for easier processing.
        lane_segment_coords = lane_segment_coords.reshape(-1, 2)
        lane_segment_coords = _transform_to_relative_frame(lane_segment_coords, anchor_state)
        lane_segment_coords = lane_segment_coords.reshape(-1, 2, 2).astype(np.float32)

        if self._connection_scales:
            # Generate multi-scale connections.
            multi_scale_connections = _generate_multi_scale_connections(
                lane_segment_conns, self._connection_scales
            )
        else:
            # Use the 1-hop connections if connection_scales is not specified.
            multi_scale_connections = {1: lane_segment_conns}

        return VectorMap(
            coords=[lane_segment_coords],
            multi_scale_connections=[multi_scale_connections],
        )
