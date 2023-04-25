from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type

import torch

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.geometry.torch_geometry import coordinates_to_local_frame
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.training.preprocessing.feature_builders.scriptable_feature_builder import ScriptableFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
    LaneOnRouteStatusData,
    LaneSegmentConnections,
    LaneSegmentCoords,
    LaneSegmentGroupings,
    LaneSegmentTrafficLightData,
    get_neighbor_vector_map,
    get_on_route_status,
    get_traffic_light_encoding,
)
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap


def _accumulate_connections(
    node_idx_to_neighbor_dict: Dict[int, Dict[str, Dict[int, int]]], scales: List[int]
) -> Dict[int, torch.Tensor]:
    """
    Accumulate the connections over multiple scales
    :param node_idx_to_neighbor_dict: {node_idx: neighbor_dict} where each neighbor_dict
                                      will have format {'i_hop_neighbors': set_of_i_hop_neighbors}
    :param scales: Connections scales to generate.
    :return: Multi-scale connections as a dict of {scale: connections_of_scale}.
    """
    # Get the connections of each scale.
    multi_scale_connections: Dict[int, torch.Tensor] = {}
    for scale in scales:
        scale_hop_neighbors = f"{scale}_hop_neighbors"

        scale_connections: List[List[int]] = []
        for node_idx, neighbor_dict in node_idx_to_neighbor_dict.items():
            for n_hop_neighbor in neighbor_dict[scale_hop_neighbors]:
                scale_connections.append([node_idx, n_hop_neighbor])

        # if cannot find n-hop neighbors, return empty conqnection with size [0,2]
        if len(scale_connections) == 0:
            multi_scale_connections[scale] = torch.empty((0, 2), dtype=torch.int64)
        else:
            multi_scale_connections[scale] = torch.tensor(scale_connections, dtype=torch.int64)

    return multi_scale_connections


def _generate_multi_scale_connections(connections: torch.Tensor, scales: List[int]) -> Dict[int, torch.Tensor]:
    """
    Generate multi-scale connections by finding the neighbors up to max(scales) hops away for each node.
    :param connections: <torch.Tensor: num_connections, 2>. A 1-hop connection is represented by [start_idx, end_idx]
    :param scales: Connections scales to generate.
    :return: Multi-scale connections as a dict of {scale: connections_of_scale}.
             Each connections_of_scale is represented by an array of <np.ndarray: num_connections, 2>,
    """
    if len(connections.shape) != 2 or connections.shape[1] != 2:
        raise ValueError(f"Unexpected connections shape: {connections.shape}")

    # This dict will have format {node_idx: neighbor_dict},
    # where each neighbor_dict will have format {'i_hop_neighbors': set_of_i_hop_neighbors}.
    # The final Dict is actually a set. But Set isn't supported by torchscript,
    #    so a filler is used for the dict's value
    node_idx_to_neighbor_dict: Dict[int, Dict[str, Dict[int, int]]] = {}
    dummy_value: int = 0

    # Initialize the data structure for each node with its 1-hop neighbors.
    for connection in connections:
        start_idx, end_idx = connection[0].item(), connection[1].item()
        if start_idx not in node_idx_to_neighbor_dict:
            start_empty: Dict[int, int] = {}
            node_idx_to_neighbor_dict[start_idx] = {"1_hop_neighbors": start_empty}
        if end_idx not in node_idx_to_neighbor_dict:
            end_empty: Dict[int, int] = {}
            node_idx_to_neighbor_dict[end_idx] = {"1_hop_neighbors": end_empty}
        node_idx_to_neighbor_dict[start_idx]["1_hop_neighbors"][end_idx] = dummy_value

    # Find the neighbors up to max(scales) hops away for each node.
    for scale in range(2, max(scales) + 1):
        scale_hop_neighbors = f"{scale}_hop_neighbors"
        prev_scale_hop_neighbors = f"{scale - 1}_hop_neighbors"

        for neighbor_dict in node_idx_to_neighbor_dict.values():
            empty: Dict[int, int] = {}
            neighbor_dict[scale_hop_neighbors] = empty
            for n_hop_neighbor in neighbor_dict[prev_scale_hop_neighbors]:
                for n_plus_1_hop_neighbor in node_idx_to_neighbor_dict[n_hop_neighbor]["1_hop_neighbors"]:
                    neighbor_dict[scale_hop_neighbors][n_plus_1_hop_neighbor] = dummy_value

    return _accumulate_connections(node_idx_to_neighbor_dict, scales)


class VectorMapFeatureBuilder(ScriptableFeatureBuilder):
    """
    Feature builder for constructing map features in a vector-representation.
    """

    def __init__(self, radius: float, connection_scales: Optional[List[int]] = None) -> None:
        """
        Initialize vector map builder with configuration parameters.
        :param radius:  The query radius scope relative to the current ego-pose.
        :param connection_scales: Connection scales to generate. Use the 1-hop connections if it's left empty.
        :return: Vector map data including lane segment coordinates and connections within the given range.
        """
        super().__init__()
        self._radius = radius
        self._connection_scales = connection_scales

    @torch.jit.unused
    def get_feature_type(self) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return VectorMap  # type: ignore

    @torch.jit.unused
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "vector_map"

    @torch.jit.unused
    def get_features_from_scenario(self, scenario: AbstractScenario) -> VectorMap:
        """Inherited, see superclass."""
        with torch.no_grad():
            ego_state = scenario.initial_ego_state
            ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
            (
                lane_seg_coords,
                lane_seg_conns,
                lane_seg_groupings,
                lane_seg_lane_ids,
                lane_seg_roadblock_ids,
            ) = get_neighbor_vector_map(scenario.map_api, ego_coords, self._radius)

            # compute route following status
            on_route_status = get_on_route_status(scenario.get_route_roadblock_ids(), lane_seg_roadblock_ids)

            # get traffic light status
            traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))
            traffic_light_data = get_traffic_light_encoding(lane_seg_lane_ids, traffic_light_data)

            tensors, list_tensors, list_list_tensors = self._pack_to_feature_tensor_dict(
                lane_seg_coords,
                lane_seg_conns,
                lane_seg_groupings,
                on_route_status,
                traffic_light_data,
                ego_state.rear_axle,
            )

            tensor_data, list_tensor_data, list_list_tensor_data = self.scriptable_forward(
                tensors, list_tensors, list_list_tensors
            )

            return self._unpack_feature_from_tensor_dict(tensor_data, list_tensor_data, list_list_tensor_data)

    @torch.jit.unused
    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> VectorMap:
        """Inherited, see superclass."""
        with torch.no_grad():
            ego_state = current_input.history.ego_states[-1]
            ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
            (
                lane_seg_coords,
                lane_seg_conns,
                lane_seg_groupings,
                lane_seg_lane_ids,
                lane_seg_roadblock_ids,
            ) = get_neighbor_vector_map(initialization.map_api, ego_coords, self._radius)

            # compute route following status
            on_route_status = get_on_route_status(initialization.route_roadblock_ids, lane_seg_roadblock_ids)

            # get traffic light status
            if current_input.traffic_light_data is None:
                raise ValueError("Cannot build VectorMap feature. PlannerInput.traffic_light_data is None")

            traffic_light_data = current_input.traffic_light_data
            traffic_light_data = get_traffic_light_encoding(lane_seg_lane_ids, traffic_light_data)

            tensors, list_tensors, list_list_tensors = self._pack_to_feature_tensor_dict(
                lane_seg_coords,
                lane_seg_conns,
                lane_seg_groupings,
                on_route_status,
                traffic_light_data,
                ego_state.rear_axle,
            )

            tensor_data, list_tensor_data, list_list_tensor_data = self.scriptable_forward(
                tensors, list_tensors, list_list_tensors
            )

            return self._unpack_feature_from_tensor_dict(tensor_data, list_tensor_data, list_list_tensor_data)

    @torch.jit.ignore
    def _unpack_feature_from_tensor_dict(
        self,
        tensor_data: Dict[str, torch.Tensor],
        list_tensor_data: Dict[str, List[torch.Tensor]],
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]],
    ) -> VectorMap:
        """
        Unpacks the data returned from the scriptable portion of the method into a VectorMap object.
        :param tensor_data: The tensor data to unpack.
        :param list_tensor_data: The List[tensor] data to unpack.
        :param list_list_tensor_data: The List[List[tensor]] data to unpack.
        :return: The unpacked VectorMap.
        """
        # Rebuild the multi scale connections from the list data.
        # Also convert tensors back to numpy arrays that are expected by other pipeline components
        #   (augmentators, etc.)
        # The feature builder outputs batch sizes of 1, so the batch dimension for the conversion is always [0].
        multi_scale_connections: Dict[int, torch.Tensor] = {}
        for key in list_tensor_data:
            if key.startswith("vector_map.multi_scale_connections_"):
                multi_scale_connections[int(key[len("vector_map.multi_scale_connections_") :])] = (
                    list_tensor_data[key][0].detach().numpy()
                )

        lane_groupings = [t.detach().numpy() for t in list_list_tensor_data["vector_map.lane_groupings"][0]]

        return VectorMap(
            coords=[list_tensor_data["vector_map.coords"][0].detach().numpy()],
            lane_groupings=[lane_groupings],
            multi_scale_connections=[multi_scale_connections],
            on_route_status=[list_tensor_data["vector_map.on_route_status"][0].detach().numpy()],
            traffic_light_data=[list_tensor_data["vector_map.traffic_light_data"][0].detach().numpy()],
        )

    @torch.jit.ignore
    def _pack_to_feature_tensor_dict(
        self,
        lane_coords: LaneSegmentCoords,
        lane_conns: LaneSegmentConnections,
        lane_groupings: LaneSegmentGroupings,
        lane_on_route_status: LaneOnRouteStatusData,
        traffic_light_data: LaneSegmentTrafficLightData,
        anchor_state: StateSE2,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Transforms the provided map and actor state primitives into scriptable types.
        This is to prepare for the scriptable portion of the feature tranform.
        :param lane_coords: The LaneSegmentCoords returned from `get_neighbor_vector_map` to transform.
        :param lane_conns: The LaneSegmentConnections returned from `get_neighbor_vector_map` to transform.
        :param lane_groupings: The LaneSegmentGroupings returned from `get_neighbor_vector_map` to transform.
        :param lane_on_route_status: The LaneOnRouteStatusData returned from `get_neighbor_vector_map` to transform.
        :param traffic_light_data: The LaneSegmentTrafficLightData returned from `get_neighbor_vector_map` to transform.
        :param anchor_state: The ego state to transform to vector.
        """
        lane_segment_coords: torch.tensor = torch.tensor(lane_coords.to_vector(), dtype=torch.float64)
        lane_segment_conns: torch.tensor = torch.tensor(lane_conns.to_vector(), dtype=torch.int64)
        on_route_status: torch.tensor = torch.tensor(lane_on_route_status.to_vector(), dtype=torch.float32)
        traffic_light_array: torch.tensor = torch.tensor(traffic_light_data.to_vector(), dtype=torch.float32)
        lane_segment_groupings: List[torch.tensor] = []

        for lane_grouping in lane_groupings.to_vector():
            lane_segment_groupings.append(torch.tensor(lane_grouping, dtype=torch.int64))

        anchor_state_tensor = torch.tensor([anchor_state.x, anchor_state.y, anchor_state.heading], dtype=torch.float64)

        return (
            {
                "lane_segment_coords": lane_segment_coords,
                "lane_segment_conns": lane_segment_conns,
                "on_route_status": on_route_status,
                "traffic_light_array": traffic_light_array,
                "anchor_state": anchor_state_tensor,
            },
            {"lane_segment_groupings": lane_segment_groupings},
            {},
        )

    @torch.jit.export
    def scriptable_forward(
        self,
        tensor_data: Dict[str, torch.Tensor],
        list_tensor_data: Dict[str, List[torch.Tensor]],
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Implemented. See interface.
        """
        lane_segment_coords = tensor_data["lane_segment_coords"]
        anchor_state = tensor_data["anchor_state"]
        lane_segment_conns = tensor_data["lane_segment_conns"]

        # lane_segment_conns can be an empty tensor.
        # In that case, the tensor will have a size of [0].
        # This will break future code, which assumes a size of [N, 2]
        # Add the extra dimension.
        if len(lane_segment_conns.shape) == 1:
            if lane_segment_conns.shape[0] == 0:
                lane_segment_conns = torch.zeros(
                    (0, 2), device=lane_segment_coords.device, layout=lane_segment_coords.layout, dtype=torch.int64
                )
            else:
                raise ValueError(f"Unexpected shape for lane_segment_conns: {lane_segment_conns.shape}")

        # Transform the lane coordinates from global frame to ego vehicle frame.
        # Flatten lane_segment_coords from (num_lane_segment, 2, 2) to (num_lane_segment * 2, 2) for easier processing.
        lane_segment_coords = lane_segment_coords.reshape(-1, 2)
        lane_segment_coords = coordinates_to_local_frame(lane_segment_coords, anchor_state, precision=torch.float64)
        lane_segment_coords = lane_segment_coords.reshape(-1, 2, 2).float()

        if self._connection_scales is not None:
            # Generate multi-scale connections.
            multi_scale_connections = _generate_multi_scale_connections(lane_segment_conns, self._connection_scales)
        else:
            multi_scale_connections = {1: lane_segment_conns}

        # Reshape all created tensors to match the dimensions and datatypes in VectorMap.
        #  (e.g. all tensors are wrapped in an additional list for batch collation)
        list_list_tensor_output: Dict[str, List[List[torch.Tensor]]] = {
            "vector_map.lane_groupings": [list_tensor_data["lane_segment_groupings"]],
        }

        list_tensor_output: Dict[str, List[torch.Tensor]] = {
            "vector_map.coords": [lane_segment_coords],
            "vector_map.on_route_status": [tensor_data["on_route_status"]],
            "vector_map.traffic_light_data": [tensor_data["traffic_light_array"]],
        }

        for key in multi_scale_connections:
            list_tensor_output[f"vector_map.multi_scale_connections_{key}"] = [multi_scale_connections[key]]

        tensor_output: Dict[str, torch.Tensor] = {}

        return tensor_output, list_tensor_output, list_list_tensor_output

    @torch.jit.export
    def precomputed_feature_config(self) -> Dict[str, Dict[str, str]]:
        """
        Implemented. See Interface.
        """
        empty: Dict[str, str] = {}
        return {"neighbor_vector_map": {"radius": str(self._radius)}, "initial_ego_state": empty}
