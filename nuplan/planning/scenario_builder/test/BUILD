load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "mock_abstract_scenario",
    srcs = ["mock_abstract_scenario.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:tracked_objects",
        "//nuplan/common/actor_state/test:test_utils",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps:abstract_map_factory",
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/database/utils/boxes:box3d",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/scenario_builder:scenario_utils",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario",
        "//nuplan/planning/simulation/history:simulation_history_buffer",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/planner:simple_planner",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
        "//nuplan/planning/simulation/trajectory:trajectory_sampling",
    ],
)

py_library(
    name = "mock_abstract_scenario_builder",
    srcs = ["mock_abstract_scenario_builder.py"],
    deps = [
        "//nuplan/common/maps:abstract_map",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/scenario_builder:abstract_scenario_builder",
        "//nuplan/planning/scenario_builder:scenario_filter",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
        "//nuplan/planning/utils/multithreading:worker_pool",
    ],
)

py_test(
    name = "test_scenario_utils",
    size = "medium",
    srcs = ["test_scenario_utils.py"],
    deps = [
        "//nuplan/planning/scenario_builder:scenario_utils",
        requirement("numpy"),
    ],
)
