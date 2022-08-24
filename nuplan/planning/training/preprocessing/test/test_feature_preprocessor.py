import logging
import pathlib
import unittest
from typing import Any

import mock

from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.planning.scenario_builder.nuplan_db.test.nuplan_scenario_test_utils import get_test_nuplan_scenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.agents_feature_builder import AgentsFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.raster_feature_builder import RasterFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import VectorMapFeatureBuilder
from nuplan.planning.training.preprocessing.feature_preprocessor import FeaturePreprocessor
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)


class TestFeaturePreprocessor(unittest.TestCase):
    """Tests preprocessing and caching functionality during training."""

    def setUp(self) -> None:
        """
        Set up test case.
        """
        self.cache_path = pathlib.Path("/tmp/test")

    def test_sample(self) -> None:
        """
        Test computation of a features for sample
        """
        # Features
        raster_feature_builder = RasterFeatureBuilder(
            map_features={'LANE': 1.0, 'INTERSECTION': 1.0, 'STOP_LINE': 0.5, 'CROSSWALK': 0.5},
            num_input_channels=4,
            target_width=224,
            target_height=224,
            target_pixel_size=0.5,
            ego_width=2.297,
            ego_front_length=4.049,
            ego_rear_length=1.127,
            ego_longitudinal_offset=0.0,
            baseline_path_thickness=1,
        )
        vectormap_builder = VectorMapFeatureBuilder(radius=20)

        # Targets
        ego_trajectory_target_builder = EgoTrajectoryTargetBuilder(TrajectorySampling(num_poses=10, time_horizon=5.0))

        logging.basicConfig(level=logging.INFO)

        feature_preprocessor = FeaturePreprocessor(
            cache_path=str(self.cache_path),
            feature_builders=[raster_feature_builder, vectormap_builder],
            target_builders=[ego_trajectory_target_builder],
            force_feature_computation=False,
        )

        # Extract scenario
        scenario = get_test_nuplan_scenario()

        self._compute_features_and_check_builders(scenario, feature_preprocessor, 2, 1)

    @mock.patch(
        'nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario.NuPlanScenario.get_tracked_objects_at_iteration'
    )
    def test_invalid_feature(self, mock_tracked_objects_fn: mock.Mock) -> None:
        """Test that empty features are invalidated."""
        agents_builder = AgentsFeatureBuilder(TrajectorySampling(4, 1.5))
        feature_preprocessor = FeaturePreprocessor(
            cache_path=str(self.cache_path),
            feature_builders=[agents_builder],
            target_builders=[],
            force_feature_computation=False,
        )

        mock_tracked_objects_fn.return_value = DetectionsTracks(TrackedObjects([]))  # scenario has no agents
        scenario = get_test_nuplan_scenario()
        features, _, _ = feature_preprocessor.compute_features(scenario)

        self.assertFalse(features['agents'].is_valid)

    def _compute_features_and_check_builders(
        self, sample: Any, feature_preprocessor: FeaturePreprocessor, number_of_features: int, number_of_targets: int
    ) -> None:
        """
        :param sample: Input data sample to compute features/targets from.
        :param feature_preprocessor: Preprocessor object with caching mechanism.
        :param number_of_features: Number of expected features.
        :param number_of_targets: Number of expected targets.
        """
        features, targets, _ = feature_preprocessor.compute_features(sample)
        self.assertEqual(len(targets), number_of_targets)
        self.assertEqual(len(features), number_of_features)

        # Validate Features
        for builder in feature_preprocessor.feature_builders:
            self.assertTrue(builder.get_feature_unique_name() in features.keys())
            feature = features[builder.get_feature_unique_name()]
            self.assertIsInstance(feature, builder.get_feature_type())
            self.assertTrue(feature.is_valid)

        # Validate Targets
        for builder in feature_preprocessor.target_builders:
            self.assertTrue(builder.get_feature_unique_name() in targets.keys())
            target = targets[builder.get_feature_unique_name()]
            self.assertIsInstance(target, builder.get_feature_type())
            self.assertTrue(target.is_valid)


if __name__ == '__main__':
    unittest.main()
