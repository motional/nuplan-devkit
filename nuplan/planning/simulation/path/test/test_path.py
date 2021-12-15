import unittest

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.path.interpolated_path import InterpolatedPath
from nuplan.planning.simulation.path.utils import calculate_progress, convert_se2_path_to_progress_path, \
    get_trimmed_path_up_to_progress


class TestPathUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.path = [StateSE2(0, 0, 0), StateSE2(3, 4, 1), StateSE2(7, 7, 2), StateSE2(10, 10, 3)]

    def test_calculate_progress(self) -> None:
        """ Tests if progress is calculated correctly """
        progress = calculate_progress(self.path)
        self.assertEqual([0.0, 5.0, 10.0, 14.242640687119284], progress)

    def test_convert_se2_path_to_progress_path(self) -> None:
        """ Tests if conversion to List[ProgressStateSE2] is calculated correctly """
        progress_path = convert_se2_path_to_progress_path(self.path)
        self.assertEqual([0.0, 5.0, 10.0, 14.242640687119284], [point.progress for point in progress_path])
        self.assertEqual(self.path, [StateSE2(x=point.x, y=point.y, heading=point.heading) for point in progress_path])


class TestInterpolatedPath(unittest.TestCase):
    def setUp(self) -> None:
        self.path = [StateSE2(0, 0, 0), StateSE2(3, 4, 1), StateSE2(7, 7, 2), StateSE2(10, 10, 3)]
        self.interpolated_path = InterpolatedPath(convert_se2_path_to_progress_path(self.path))

    def test_get_start_progress(self) -> None:
        """ Check start progress """
        self.assertEqual(self.interpolated_path.get_start_progress(),
                         self.interpolated_path.get_sampled_path()[0].progress)

    def test_get_end_progress(self) -> None:
        """ Check end progress """
        self.assertEqual(self.interpolated_path.get_end_progress(),
                         self.interpolated_path.get_sampled_path()[-1].progress)

    def test_get_state_at_progress(self) -> None:
        """ Check if the interpolated states are calculated correctly progress """
        state = self.interpolated_path.get_state_at_progress(5)
        self.assertEqual(5, state.progress)
        self.assertEqual(3, state.x)
        self.assertEqual(4, state.y)
        self.assertEqual(1, state.heading)

    def test_get_state_at_progress_expect_throw(self) -> None:
        """ Check if assertion is raised for invalid calls """
        self.assertRaises(AssertionError, self.interpolated_path.get_state_at_progress, 100)
        self.assertRaises(AssertionError, self.interpolated_path.get_state_at_progress, -1)

    def test_get_sampled_path(self) -> None:
        sample_path = self.interpolated_path.get_sampled_path()
        self.assertEqual(self.interpolated_path._path, sample_path)
        sample_path = get_trimmed_path_up_to_progress(self.interpolated_path, 2)
        self.assertEqual([self.interpolated_path.get_state_at_progress(2)] + self.interpolated_path._path[1:],
                         sample_path)
        trimmed_sample_path = get_trimmed_path_up_to_progress(self.interpolated_path, 5)
        self.assertEqual(self.interpolated_path._path[1:], trimmed_sample_path)


if __name__ == '__main__':
    unittest.main()
