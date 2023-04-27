from __future__ import annotations

from typing import Dict, List, Tuple, Type

import torch

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.geometry.torch_geometry import vector_set_coordinates_to_local_frame
from nuplan.common.maps.maps_datatypes import TrafficLightStatuses
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.training.preprocessing.feature_builders.scriptable_feature_builder import ScriptableFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
    LaneSegmentTrafficLightData,
    MapObjectPolylines,
    VectorFeatureLayer,
    get_neighbor_vector_set_map,
)
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature, FeatureDataType
from nuplan.planning.training.preprocessing.features.vector_set_map import VectorSetMap
from nuplan.planning.training.preprocessing.utils.vector_preprocessing import convert_feature_layer_to_fixed_size


class VectorSetMapFeatureBuilder(ScriptableFeatureBuilder):
    """
    Feature builder for constructing map features in a vector set representation, similar to that of
        VectorNet ("VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation").
    """

    def __init__(
        self,
        map_features: List[str],
        max_elements: Dict[str, int],
        max_points: Dict[str, int],
        radius: float,
        interpolation_method: str,
    ) -> None:
        """
        Initialize vector set map builder with configuration parameters.
        :param map_features: name of map features to be extracted.
        :param max_elements: maximum number of elements to extract per feature layer.
        :param max_points: maximum number of points per feature to extract per feature layer.
        :param radius:  [m ]The query radius scope relative to the current ego-pose.
        :param interpolation_method: Interpolation method to apply when interpolating to maintain fixed size
            map elements.
        :return: Vector set map data including map element coordinates and traffic light status info.
        """
        super().__init__()
        self.map_features = map_features
        self.max_elements = max_elements
        self.max_points = max_points
        self.radius = radius
        self.interpolation_method = interpolation_method
        self._traffic_light_encoding_dim = LaneSegmentTrafficLightData.encoding_dim()

        # Sanitize feature building parameters
        for feature_name in self.map_features:
            try:
                VectorFeatureLayer[feature_name]
            except KeyError:
                raise ValueError(f"Object representation for layer: {feature_name} is unavailable!")
            if feature_name not in self.max_elements:
                raise RuntimeError(f"Max elements unavailable for {feature_name} feature layer!")
            if feature_name not in self.max_points:
                raise RuntimeError(f"Max points unavailable for {feature_name} feature layer!")

    @torch.jit.unused
    def get_feature_type(self) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return VectorSetMap  # type: ignore

    @torch.jit.unused
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "vector_set_map"

    @torch.jit.unused
    def get_scriptable_input_from_scenario(
        self, scenario: AbstractScenario
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Extract the input for the scriptable forward method from the scenario object
        :param scenario: planner input from training
        :returns: Tensor data + tensor list data to be used in scriptable forward
        """
        ego_state = scenario.initial_ego_state
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        route_roadblock_ids = scenario.get_route_roadblock_ids()
        traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))

        coords, traffic_light_data = get_neighbor_vector_set_map(
            scenario.map_api,
            self.map_features,
            ego_coords,
            self.radius,
            route_roadblock_ids,
            [TrafficLightStatuses(traffic_light_data)],
        )

        tensor, list_tensor, list_list_tensor = self._pack_to_feature_tensor_dict(
            coords, traffic_light_data[0], ego_state.rear_axle
        )
        return tensor, list_tensor, list_list_tensor

    @torch.jit.unused
    def get_scriptable_input_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Extract the input for the scriptable forward method from the simulation objects
        :param current_input: planner input from sim
        :param initialization: planner initialization from sim
        :returns: Tensor data + tensor list data to be used in scriptable forward
        """
        ego_state = current_input.history.ego_states[-1]
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        route_roadblock_ids = initialization.route_roadblock_ids

        # get traffic light status
        if current_input.traffic_light_data is None:
            raise ValueError("Cannot build VectorSetMap feature. PlannerInput.traffic_light_data is None")
        traffic_light_data = current_input.traffic_light_data

        coords, traffic_light_data = get_neighbor_vector_set_map(
            initialization.map_api,
            self.map_features,
            ego_coords,
            self.radius,
            route_roadblock_ids,
            [TrafficLightStatuses(traffic_light_data)],
        )

        tensor, list_tensor, list_list_tensor = self._pack_to_feature_tensor_dict(
            coords, traffic_light_data[0], ego_state.rear_axle
        )
        return tensor, list_tensor, list_list_tensor

    @torch.jit.unused
    def get_features_from_scenario(self, scenario: AbstractScenario) -> VectorSetMap:
        """Inherited, see superclass."""
        tensor_data, list_tensor_data, list_list_tensor_data = self.get_scriptable_input_from_scenario(scenario)
        tensor_data, list_tensor_data, list_list_tensor_data = self.scriptable_forward(
            tensor_data, list_tensor_data, list_list_tensor_data
        )

        return self._unpack_feature_from_tensor_dict(tensor_data, list_tensor_data, list_list_tensor_data)

    @torch.jit.unused
    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> VectorSetMap:
        """Inherited, see superclass."""
        tensor_data, list_tensor_data, list_list_tensor_data = self.get_scriptable_input_from_simulation(
            current_input, initialization
        )
        tensor_data, list_tensor_data, list_list_tensor_data = self.scriptable_forward(
            tensor_data, list_tensor_data, list_list_tensor_data
        )

        return self._unpack_feature_from_tensor_dict(tensor_data, list_tensor_data, list_list_tensor_data)

    @torch.jit.unused
    def _unpack_feature_from_tensor_dict(
        self,
        tensor_data: Dict[str, torch.Tensor],
        list_tensor_data: Dict[str, List[torch.Tensor]],
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]],
    ) -> VectorSetMap:
        """
        Unpacks the data returned from the scriptable portion of the method into a VectorSetMap object.
        :param tensor_data: The tensor data to unpack.
        :param list_tensor_data: The List[tensor] data to unpack.
        :param list_list_tensor_data: The List[List[tensor]] data to unpack.
        :return: The unpacked VectorSetMap.
        """
        coords: Dict[str, List[FeatureDataType]] = {}
        traffic_light_data: Dict[str, List[FeatureDataType]] = {}
        availabilities: Dict[str, List[FeatureDataType]] = {}

        for key in list_tensor_data:
            if key.startswith("vector_set_map.coords."):
                feature_name = key[len("vector_set_map.coords.") :]
                coords[feature_name] = [list_tensor_data[key][0].detach().numpy()]
            if key.startswith("vector_set_map.traffic_light_data."):
                feature_name = key[len("vector_set_map.traffic_light_data.") :]
                traffic_light_data[feature_name] = [list_tensor_data[key][0].detach().numpy()]
            if key.startswith("vector_set_map.availabilities."):
                feature_name = key[len("vector_set_map.availabilities.") :]
                availabilities[feature_name] = [list_tensor_data[key][0].detach().numpy()]

        return VectorSetMap(
            coords=coords,
            traffic_light_data=traffic_light_data,
            availabilities=availabilities,
        )

    @torch.jit.unused
    def _pack_to_feature_tensor_dict(
        self,
        coords: Dict[str, MapObjectPolylines],
        traffic_light_data: Dict[str, LaneSegmentTrafficLightData],
        anchor_state: StateSE2,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Transforms the provided map and actor state primitives into scriptable types.
        This is to prepare for the scriptable portion of the feature transform.
        :param coords: Dictionary mapping feature name to polyline vector sets.
        :param traffic_light_data: Dictionary mapping feature name to traffic light info corresponding to map elements
            in coords.
        :param anchor_state: The ego state to transform to vector.
        :return
           tensor_data: Packed tensor data.
           list_tensor_data: Packed List[tensor] data.
           list_list_tensor_data: Packed List[List[tensor]] data.
        """
        tensor_data: Dict[str, torch.Tensor] = {}
        # anchor state
        anchor_state_tensor = torch.tensor([anchor_state.x, anchor_state.y, anchor_state.heading], dtype=torch.float64)
        tensor_data["anchor_state"] = anchor_state_tensor

        list_tensor_data: Dict[str, List[torch.Tensor]] = {}

        for feature_name, feature_coords in coords.items():
            list_feature_coords: List[torch.Tensor] = []

            # Pack coords into tensor list
            for element_coords in feature_coords.to_vector():
                list_feature_coords.append(torch.tensor(element_coords, dtype=torch.float64))
            list_tensor_data[f"coords.{feature_name}"] = list_feature_coords

            # Pack traffic light data into tensor list if it exists
            if feature_name in traffic_light_data:
                list_feature_tl_data: List[torch.Tensor] = []

                for element_tl_data in traffic_light_data[feature_name].to_vector():
                    list_feature_tl_data.append(torch.tensor(element_tl_data, dtype=torch.float32))
                list_tensor_data[f"traffic_light_data.{feature_name}"] = list_feature_tl_data

        return (
            tensor_data,
            list_tensor_data,
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
        tensor_output: Dict[str, torch.Tensor] = {}
        list_tensor_output: Dict[str, List[torch.Tensor]] = {}
        list_list_tensor_output: Dict[str, List[List[torch.Tensor]]] = {}

        anchor_state = tensor_data["anchor_state"]

        for feature_name in self.map_features:
            if f"coords.{feature_name}" in list_tensor_data:
                feature_coords = list_tensor_data[f"coords.{feature_name}"]
                feature_tl_data = (
                    [list_tensor_data[f"traffic_light_data.{feature_name}"]]
                    if f"traffic_light_data.{feature_name}" in list_tensor_data
                    else None
                )

                coords, tl_data, avails = convert_feature_layer_to_fixed_size(
                    feature_coords,
                    feature_tl_data,
                    self.max_elements[feature_name],
                    self.max_points[feature_name],
                    self._traffic_light_encoding_dim,
                    interpolation=self.interpolation_method  # apply interpolation only for lane features
                    if feature_name
                    in [
                        VectorFeatureLayer.LANE.name,
                        VectorFeatureLayer.LEFT_BOUNDARY.name,
                        VectorFeatureLayer.RIGHT_BOUNDARY.name,
                        VectorFeatureLayer.ROUTE_LANES.name,
                    ]
                    else None,
                )

                coords = vector_set_coordinates_to_local_frame(coords, avails, anchor_state)

                list_tensor_output[f"vector_set_map.coords.{feature_name}"] = [coords]
                list_tensor_output[f"vector_set_map.availabilities.{feature_name}"] = [avails]

                if tl_data is not None:
                    list_tensor_output[f"vector_set_map.traffic_light_data.{feature_name}"] = [tl_data[0]]

        return tensor_output, list_tensor_output, list_list_tensor_output

    @torch.jit.export
    def precomputed_feature_config(self) -> Dict[str, Dict[str, str]]:
        """
        Implemented. See Interface.
        """
        empty: Dict[str, str] = {}
        max_elements: List[str] = [
            f"{feature_name}.{feature_max_elements}" for feature_name, feature_max_elements in self.max_elements.items()
        ]
        max_points: List[str] = [
            f"{feature_name}.{feature_max_points}" for feature_name, feature_max_points in self.max_points.items()
        ]

        return {
            "neighbor_vector_set_map": {
                "radius": str(self.radius),
                "interpolation_method": self.interpolation_method,
                "map_features": ",".join(self.map_features),
                "max_elements": ",".join(max_elements),
                "max_points": ",".join(max_points),
            },
            "initial_ego_state": empty,
        }
