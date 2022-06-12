import unittest

import numpy as np

from nuplan.planning.training.callbacks.utils.visualization_utils import (
    get_raster_from_vector_map_with_agents,
    get_raster_with_trajectories_as_rgb,
)
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.features.raster import Raster
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap


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
        assert image.shape == (size, size, 3)

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
        assert image.shape == (size, size, 3)


if __name__ == '__main__':
    unittest.main()
