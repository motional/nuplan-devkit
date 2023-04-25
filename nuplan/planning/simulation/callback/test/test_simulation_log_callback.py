import pathlib
import tempfile
import unittest
from typing import Any, Callable, Iterable
from unittest.mock import Mock

import numpy as np

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.callback.simulation_log_callback import SimulationLogCallback
from nuplan.planning.simulation.controller.abstract_controller import AbstractEgoController
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.simple_planner import SimplePlanner
from nuplan.planning.simulation.simulation_log import SimulationLog
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.simulation_time_controller.abstract_simulation_time_controller import (
    AbstractSimulationTimeController,
)
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


def callable_name_matches(a: Callable[..., Any], b: Callable[..., Any]) -> bool:
    """
    Checks that callable names match.
    :param a: first callable to compare.
    :param b: second callable to compare.
    :return: true if the names match, otherwise false.
    """
    # Ideally we'd use __name__, but it's not guaranteed to exist for all callables
    if hasattr(a, "__name__"):
        if a.__name__ != b.__name__:
            return False

    # If __name__ does not exist, then try to parse from repr
    # We expect the repr to look like:
    # <scipy.interpolate._interpolate.interp1d object at 0x7f86ffcd4400>
    # From this, extract scipy.interpolate._interpolate.interp1d
    elif "object at" in (a_repr := repr(a)):
        address_ind = a_repr.index("object at")
        a_name = a_repr[1 : address_ind - 1]
        b_name = repr(b)[1 : address_ind - 1]

        if a_name != b_name:
            return False

    else:
        # Don't expect to reach here, but there may be uncovered edgecases in general
        raise NotImplementedError

    return True


def iterator_is_equal(a: Iterable[Any], b: Iterable[Any]) -> bool:
    """
    Checks that two iterables are equal by value.
    :param a: a in a == b.
    :param b: b in a == b.
    :return: true if the iterable contents match.
    """
    for a_item, b_item in zip(a_iter := iter(a), b_iter := iter(b)):
        if not objects_are_equal(a_item, b_item):
            return False

    # Check that the iterator lengths match by making sure both iterators is exhausted
    try:
        next(a_iter)
        return False  # If it succeeds, the lengths didn't match
    except StopIteration:
        try:
            next(b_iter)
            return False
        except StopIteration:
            return True


def objects_are_equal(a: object, b: object) -> bool:
    """
    Recursively checks if two objects are equal by value.

    This method supports objects that are compositions of:
        * built-in types (int, float, bool, etc)
        * callable objects
        * numpy arrays
        * objects supporting `__dict__`
        * compositions of the above objects

    Other types are currently unsupported.

    :param a: a in a == b, must implement __dict__ or be directly comparable.
    :param b: b in a == b, must implement __dict__ or be directly comparable.
    :return: true if both objects are the same, otherwise false.
    """
    if not hasattr(a, "__dict__") and not hasattr(b, "__dict__"):
        return a == b

    a_dict = a.__dict__
    b_dict = b.__dict__

    if set(a_dict.keys()) != set(b_dict.keys()):
        return False

    for key in a_dict:
        if type(a_dict[key]) != type(b_dict[key]):  # noqa: E721
            return False

        # The order of the checks matters (eg. callables have __dict__, __dict__ often has __iter__)
        if callable(a_dict[key]):
            if not callable_name_matches(a_dict[key], b_dict[key]):
                return False
        elif hasattr(a_dict[key], "__dict__"):
            if not objects_are_equal(a_dict[key], b_dict[key]):
                return False
        elif isinstance(a_dict[key], np.ndarray):
            if not np.allclose(a_dict[key], b_dict[key]):
                return False
        elif hasattr(a_dict[key], "__iter__"):
            if not iterator_is_equal(a_dict[key], b_dict[key]):
                return False
        else:
            return objects_are_equal(a_dict[key], b_dict[key])
    return True


class TestSimulationLogCallback(unittest.TestCase):
    """Tests simulation_log_callback."""

    def setUp(self) -> None:
        """Setup Mocked classes."""
        self.output_folder = tempfile.TemporaryDirectory()
        self.callback = SimulationLogCallback(
            output_directory=self.output_folder.name, simulation_log_dir='simulation_log', serialization_type='msgpack'
        )

        self.sim_manager = Mock(spec=AbstractSimulationTimeController)
        self.observation = Mock(spec=AbstractObservation)
        self.controller = Mock(spec=AbstractEgoController)

    def tearDown(self) -> None:
        """Clean up folder."""
        self.output_folder.cleanup()

    def test_callback(self) -> None:
        """
        Tests whether a scene can be dumped into a simulation log, checks that the keys are correct,
        and checks that the log contains the expected data after being re-loaded from disk.
        """
        scenario = MockAbstractScenario()

        self.setup = SimulationSetup(
            observations=self.observation,
            scenario=scenario,
            time_controller=self.sim_manager,
            ego_controller=self.controller,
        )

        planner = SimplePlanner(2, 0.5, [0, 0])

        # Make sure the directory is correct
        directory = self.callback._get_scenario_folder(planner.name(), scenario)
        self.assertEqual(
            str(directory),
            self.output_folder.name
            + "/simulation_log/SimplePlanner/mock_scenario_type/mock_log_name/mock_scenario_name",
        )

        # initialize callback
        self.callback.on_initialization_start(self.setup, planner)

        # Mock two iteration steps
        history = SimulationHistory(scenario.map_api, scenario.get_mission_goal())
        state_0 = EgoState.build_from_rear_axle(
            StateSE2(0, 0, 0),
            vehicle_parameters=scenario.ego_vehicle_parameters,
            rear_axle_velocity_2d=StateVector2D(x=0, y=0),
            rear_axle_acceleration_2d=StateVector2D(x=0, y=0),
            tire_steering_angle=0,
            time_point=TimePoint(0),
        )
        state_1 = EgoState.build_from_rear_axle(
            StateSE2(0, 0, 0),
            vehicle_parameters=scenario.ego_vehicle_parameters,
            rear_axle_velocity_2d=StateVector2D(x=0, y=0),
            rear_axle_acceleration_2d=StateVector2D(x=0, y=0),
            tire_steering_angle=0,
            time_point=TimePoint(1000),
        )
        history.add_sample(
            SimulationHistorySample(
                iteration=SimulationIteration(time_point=TimePoint(0), index=0),
                ego_state=state_0,
                trajectory=InterpolatedTrajectory(trajectory=[state_0, state_1]),
                observation=DetectionsTracks(TrackedObjects()),
                traffic_light_status=list(scenario.get_traffic_light_status_at_iteration(0)),
            )
        )
        history.add_sample(
            SimulationHistorySample(
                iteration=SimulationIteration(time_point=TimePoint(0), index=0),
                ego_state=state_1,
                trajectory=InterpolatedTrajectory(trajectory=[state_0, state_1]),
                observation=DetectionsTracks(TrackedObjects()),
                traffic_light_status=list(scenario.get_traffic_light_status_at_iteration(0)),
            )
        )

        # Simulate simulation interation loop
        for data in history.data:
            self.callback.on_step_end(self.setup, planner, data)

        # Simulate end of simulation
        self.callback.on_simulation_end(self.setup, planner, history)

        # Compressed path
        path = pathlib.Path(
            self.output_folder.name
            + "/simulation_log/SimplePlanner/mock_scenario_type/mock_log_name/mock_scenario_name/mock_scenario_name.msgpack.xz"
        )

        self.assertTrue(path.exists())
        simulation_log = SimulationLog.load_data(file_path=path)
        self.assertEqual(simulation_log.file_path, path)

        self.assertTrue(objects_are_equal(simulation_log.simulation_history, history))


if __name__ == '__main__':
    unittest.main()
