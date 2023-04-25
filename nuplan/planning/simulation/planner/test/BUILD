load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_remote_planner",
    size = "small",
    srcs = ["test_remote_planner.py"],
    deps = [
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/planner:remote_planner",
    ],
)

py_test(
    name = "test_log_future_planner",
    size = "small",
    srcs = ["test_log_future_planner.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history_buffer",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/planner:log_future_planner",
        "//nuplan/planning/simulation/planner:planner_report",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
    ],
)

py_library(
    name = "mock_idm_planner",
    srcs = ["mock_idm_planner.py"],
    deps = [
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario_builder",
        "//nuplan/planning/simulation/observation/idm:utils",
        "//nuplan/planning/simulation/path:interpolated_path",
        "//nuplan/planning/simulation/planner:abstract_idm_planner",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/trajectory:abstract_trajectory",
        "//nuplan/planning/simulation/trajectory:interpolated_trajectory",
        requirement("shapely"),
    ],
)

py_test(
    name = "test_abstract_idm_planner",
    size = "medium",
    srcs = ["test_abstract_idm_planner.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/common/actor_state:agent",
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/planning/scenario_builder/nuplan_db/test:nuplan_scenario_test_utils",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/observation/idm:idm_states",
        "//nuplan/planning/simulation/occupancy_map:abstract_occupancy_map",
        "//nuplan/planning/simulation/path",
        "//nuplan/planning/simulation/planner/test:mock_idm_planner",
    ],
)

py_test(
    name = "test_idm_planner",
    size = "medium",
    srcs = ["test_idm_planner.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/planning/scenario_builder/nuplan_db/test:nuplan_scenario_test_utils",
        "//nuplan/planning/simulation/history:simulation_history_buffer",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/planner:idm_planner",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
    ],
)
