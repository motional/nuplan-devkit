from __future__ import annotations

import itertools
import unittest
from typing import Generator, List, Tuple
from unittest.mock import Mock, patch

import numpy as np
import numpy.testing as np_test

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.test.test_utils import get_sample_ego_state
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatusType
from nuplan.common.utils.test_utils.function_validation import assert_functions_swappable
from nuplan.common.utils.test_utils.patch import patch_with_validation
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
    LaneOnRouteStatusData,
    LaneSegmentConnections,
    LaneSegmentCoords,
    LaneSegmentGroupings,
    LaneSegmentLaneIDs,
    LaneSegmentRoadBlockIDs,
    LaneSegmentTrafficLightData,
    OnRouteStatusType,
)
from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import VectorMapFeatureBuilder
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap

PATCH_PREFIX = "nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder"


def _fill_in_abstract_scenario_mock_parameters(
    scenario_mock: Mock,
    initial_ego_center_pose: StateSE2,
    initial_timestamp: int,
    traffic_light_statuses: List[TrafficLightStatusData],
    route_roadblock_ids: List[str],
) -> None:
    """
    Helper function to configure a scenario mock with required parameters for VectorMapFeatureBuilder.
    :param scenario_mock: The AbstractScenario mock that we should update.
    :param initial_ego_center_pose: Ego vehicle center pose used to construct the scenario initial_ego_state.
    :param initial_timestamp: Initial timestamp corresponding to initial_ego_state for the test scenario.
    :param traffic_light_statuses: A list of TL statuses used to determine the traffic light scene at the 0th iteration.
    :param route_roadblock_ids: The route roadblock ids for the scenario.
    """
    scenario_mock.initial_ego_state = get_sample_ego_state(center=initial_ego_center_pose, time_us=initial_timestamp)
    scenario_mock.get_route_roadblock_ids.return_value = route_roadblock_ids

    def _get_traffic_light_status_at_iteration_patch(iteration: int) -> Generator[TrafficLightStatusData, None, None]:
        """A patch to populate traffic light states for the 0th iteration only."""
        if iteration != 0:
            raise ValueError("We expect the vector map builder to only use the 0th iteration TL states.")

        yield from traffic_light_statuses

    assert_functions_swappable(
        AbstractScenario.get_traffic_light_status_at_iteration,
        _get_traffic_light_status_at_iteration_patch,
    )

    scenario_mock.get_traffic_light_status_at_iteration.side_effect = _get_traffic_light_status_at_iteration_patch


def _form_lane_segment_coords_connections_from_points(
    points: List[Point2D],
    start_lane_segment_index: int,
) -> Tuple[LaneSegmentCoords, LaneSegmentConnections]:
    """
    Helper function to take in a set of points and convert into an example set of lane segments and lane connections.
    We assume that points i and (i+1) form lane segments l_i.
    We assume lane_segment l_i connects to segment l_{i+1}.
    :param points: The list of points to form lane segments + connections from.
    :param start_lane_segment_index: This is used to label the lane segments by setting the starting value of i above.
    :return: The lane segments coordinates (start + end point) and connectivity (lane_segment_from, lane_segment_to).
    """
    segments = [(p_prev, p_next) for p_prev, p_next in zip(points[:-1], points[1:])]

    connections = [
        (start_lane_segment_index + idx, start_lane_segment_index + idx + 1) for idx in range(len(segments) - 1)
    ]

    return LaneSegmentCoords(segments), LaneSegmentConnections(connections)


def _get_neighbor_vector_map_patch(
    map_api: AbstractMap, point: Point2D, radius: float
) -> Tuple[
    LaneSegmentCoords, LaneSegmentConnections, LaneSegmentGroupings, LaneSegmentLaneIDs, LaneSegmentRoadBlockIDs
]:
    """
    A patch for get_neighbor_vector_map that uses the following dummy map for testing.
    Original function docstring:
    Extract neighbor vector map information around ego vehicle.
    :param map_api: map to perform extraction on.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about vector map query range.
    :return
        lane_seg_coords: lane_segment coords in shape of [num_lane_segment, 2, 2].
        lane_seg_conns: lane_segment connections [start_idx, end_idx] in shape of [num_connection, 2].
        lane_seg_groupings: collection of lane_segment indices in each lane in shape of
            [num_lane, num_lane_segment_in_lane].
        lane_seg_lane_ids: lane ids of segments at given index in coords in shape of [num_lane_segment 1].
        lane_seg_roadblock_ids: roadblock ids of segments at given index in coords in shape of [num_lane_segment 1].

    Dummy map setup where ls = lane_segment, lc = lane/lane connector, rb = roadblock.  Origin at the center of the map.

    ls_id    0  1  2  3  4  5
    lc_id   5000    5001   5002
            ______|_____|______

            x--x--x--x--x--x--x
    origin           O
            x--x--x--x--x--x--x
            ______|_____|______
    ls_id    6  7  8  9  10 11
    lc_id   5003    5004   5005
    rb_id   60000  70000  80000
    """
    # Generate the coordinates and connections for the top and bottom lines in the dummy map.
    top_line_points = [Point2D(x=x, y=1) for x in range(-3, 4)]
    top_line_segments_coords, top_line_segment_connections = _form_lane_segment_coords_connections_from_points(
        points=top_line_points,
        start_lane_segment_index=0,
    )

    bottom_line_points = [Point2D(x=x, y=-1) for x in range(-3, 4)]
    bottom_line_segments_coords, bottom_line_segment_connections = _form_lane_segment_coords_connections_from_points(
        points=bottom_line_points,
        start_lane_segment_index=len(top_line_segments_coords.coords),
    )

    combined_coords = LaneSegmentCoords(coords=top_line_segments_coords.coords + bottom_line_segments_coords.coords)
    combined_connections = LaneSegmentConnections(
        connections=top_line_segment_connections.connections + bottom_line_segment_connections.connections
    )

    if len(combined_coords.coords) != 12:
        raise ValueError(f"Expected 12 lane segments to match dummy map.  Got {combined_coords} instead.")

    if len(combined_connections.connections) != 10:
        raise ValueError(
            f"Expected 10 lane segment connections to match dummy map.  Got {combined_connections} instead."
        )

    # Generate groupings for the lane segments, matching the dummy map above.
    combined_lane_seg_groupings = LaneSegmentGroupings([[x, x + 1] for x in range(0, 12, 2)])

    lane_id_list = [str(x) for x in range(5000, 5006)]
    combined_lane_seg_lane_ids = LaneSegmentLaneIDs(
        [doubled_entry for doubled_entry in itertools.chain.from_iterable((entry, entry) for entry in lane_id_list)]
    )

    roadblock_id_list = ["60000", "70000", "80000"] * 2
    combined_lane_seg_roadblock_ids = LaneSegmentRoadBlockIDs(
        [
            doubled_entry
            for doubled_entry in itertools.chain.from_iterable((entry, entry) for entry in roadblock_id_list)
        ]
    )

    return (
        combined_coords,
        combined_connections,
        combined_lane_seg_groupings,
        combined_lane_seg_lane_ids,
        combined_lane_seg_roadblock_ids,
    )


class TestVectorMapFeatureBuilderLaneMetadata(unittest.TestCase):
    """Test feature builder that constructs map features in vectorized format."""

    @patch(f"{PATCH_PREFIX}.AbstractScenario", autospec=True)
    def test_vectormap_example_metadata(self, mock_abstract_scenario: Mock) -> None:
        """
        Test VectorMapFeatureBuilder
        """
        test_radius = 50.0  # [m]
        builder = VectorMapFeatureBuilder(radius=test_radius, connection_scales=None)

        # Parameters that should correspond to the dummy map defined in _get_neighbor_vector_map_patch.
        # If these entries are modified, the expected returned values must also be updated as they are manually determined.
        initial_ego_center_pose = StateSE2(x=0.0, y=0.0, heading=0.0)

        timestamp = 1000000

        traffic_light_statuses = [
            TrafficLightStatusData(status=TrafficLightStatusType.GREEN, lane_connector_id=4000, timestamp=timestamp),
            TrafficLightStatusData(status=TrafficLightStatusType.GREEN, lane_connector_id=5000, timestamp=timestamp),
            TrafficLightStatusData(status=TrafficLightStatusType.YELLOW, lane_connector_id=5002, timestamp=timestamp),
            TrafficLightStatusData(status=TrafficLightStatusType.RED, lane_connector_id=5003, timestamp=timestamp),
            TrafficLightStatusData(status=TrafficLightStatusType.RED, lane_connector_id=5005, timestamp=timestamp),
        ]

        route_roadblock_ids = ["60000", "70000"]

        # Set up the mock with the parameters above.
        _fill_in_abstract_scenario_mock_parameters(
            scenario_mock=mock_abstract_scenario,
            initial_ego_center_pose=initial_ego_center_pose,
            initial_timestamp=timestamp,
            traffic_light_statuses=traffic_light_statuses,
            route_roadblock_ids=route_roadblock_ids,
        )

        # Patch to use the dummy map and parameters above for testing.
        with patch_with_validation(
            "nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder.get_neighbor_vector_map",
            _get_neighbor_vector_map_patch,
        ):
            # Call method under test.
            vector_map_feature = builder.get_features_from_scenario(mock_abstract_scenario)

            # Check generic feature properties.
            self.assertIsInstance(vector_map_feature, VectorMap)
            self.assertEqual(vector_map_feature.num_of_batches, 1)
            self.assertTrue(vector_map_feature.is_valid)

            # Check basic lane / coordinate properties.
            self.assertEqual(vector_map_feature.num_lanes_in_sample(0), 6)
            self.assertEqual(vector_map_feature.get_lane_coords(0).shape, (12, 2, 2))

            # Check expected traffic light features vs. actual.
            actual_traffic_light_data = vector_map_feature.traffic_light_data[0]

            # Note: At the moment, only green/red TLs are populated.
            # Everything else (including yellow) is marked as unknown.
            expected_traffic_light_data = np.zeros_like(actual_traffic_light_data)
            tl_encoding_dict = LaneSegmentTrafficLightData._one_hot_encoding
            expected_traffic_light_data[[0, 1]] = tl_encoding_dict[TrafficLightStatusType.GREEN]
            expected_traffic_light_data[[6, 7, 10, 11]] = tl_encoding_dict[TrafficLightStatusType.RED]
            expected_traffic_light_data[[2, 3, 4, 5, 8, 9]] = tl_encoding_dict[TrafficLightStatusType.UNKNOWN]

            np_test.assert_allclose(actual_traffic_light_data, expected_traffic_light_data)

            # Check expected on route features vs. actual.
            actual_on_route_status = vector_map_feature.on_route_status[0]

            expected_on_route_status = np.zeros_like(actual_on_route_status)
            route_encoding_dict = LaneOnRouteStatusData._binary_encoding
            expected_on_route_status[[4, 5, 10, 11]] = route_encoding_dict[OnRouteStatusType.OFF_ROUTE]
            expected_on_route_status[[0, 1, 2, 3, 6, 7, 8, 9]] = route_encoding_dict[OnRouteStatusType.ON_ROUTE]

            np_test.assert_allclose(actual_on_route_status, expected_on_route_status)


if __name__ == '__main__':
    unittest.main()
