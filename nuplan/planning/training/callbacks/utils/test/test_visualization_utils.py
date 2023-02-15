import unittest

import numpy as np

from nuplan.planning.training.callbacks.utils.visualization_utils import (
    get_raster_from_vector_map_with_agents,
    get_raster_with_trajectories_as_rgb,
)
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import VectorFeatureLayer
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.features.generic_agents import GenericAgents
from nuplan.planning.training.preprocessing.features.raster import Raster
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap
from nuplan.planning.training.preprocessing.features.vector_set_map import VectorSetMap


class TestVisualizationUtils(unittest.TestCase):
    """Unit tests for visualization utlities."""

    def test_raster_visualization(self) -> None:
        """
        Test raster visualization utils.
        """
        trajectory_1 = Trajectory(data=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))
        trajectory_2 = Trajectory(data=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]]))

        size = 224
        raster = Raster(data=np.zeros((1, size, size, 4)))

        image = get_raster_with_trajectories_as_rgb(raster, trajectory_1, trajectory_2)

        # Check dimensions
        self.assertEqual(image.shape, (size, size, 3))
        # Check if objects are drawn on to the raster
        self.assertTrue(np.any(image))

    def test_vector_map_agents_visualization(self) -> None:
        """
        Test vector map and agents visualization utils.
        """
        trajectory_1 = Trajectory(data=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))
        trajectory_2 = Trajectory(data=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]]))

        pixel_size = 0.5
        radius = 50.0
        size = int(2 * radius / pixel_size)

        vector_map = VectorMap(
            coords=[np.zeros((1000, 2, 2))],
            lane_groupings=[[]],
            multi_scale_connections=[{}],
            on_route_status=[np.zeros((1000, 2))],
            traffic_light_data=[np.zeros((1000, 4))],
        )

        agents = Agents(
            ego=[np.array(([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))],
            agents=[
                np.array(
                    [
                        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                    ]
                )
            ],
        )

        image = get_raster_from_vector_map_with_agents(
            vector_map, agents, trajectory_1, trajectory_2, pixel_size=pixel_size, radius=radius
        )

        # Check dimensions
        self.assertEqual(image.shape, (size, size, 3))
        # Check if objects are drawn on to the raster
        self.assertTrue(np.any(image))

    def test_vector_set_map_generic_agents_visualization(self) -> None:
        """
        Test vector set map and generic agents visualization utils.
        """
        trajectory_1 = Trajectory(data=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))
        trajectory_2 = Trajectory(data=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]]))

        pixel_size = 0.5
        radius = 50.0
        size = int(2 * radius / pixel_size)
        agent_features = ['VEHICLE', 'PEDESTRIAN', 'BICYCLE']

        vector_set_map = VectorSetMap(
            coords={VectorFeatureLayer.LANE.name: [np.zeros((100, 100, 2))]},
            traffic_light_data={},
            availabilities={VectorFeatureLayer.LANE.name: [np.ones((100, 100), dtype=bool)]},
        )

        agents = GenericAgents(
            ego=[np.array(([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))],
            agents={
                feature_name: [
                    np.array(
                        [
                            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                        ]
                    )
                ]
                for feature_name in agent_features
            },
        )

        image = get_raster_from_vector_map_with_agents(
            vector_set_map, agents, trajectory_1, trajectory_2, pixel_size=pixel_size, radius=radius
        )

        # Check dimensions
        self.assertEqual(image.shape, (size, size, 3))
        # Check if objects are drawn on to the raster
        self.assertTrue(np.any(image))


if __name__ == '__main__':
    unittest.main()
