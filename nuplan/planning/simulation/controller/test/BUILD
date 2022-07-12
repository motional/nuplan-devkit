load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_log_playback",
    size = "small",
    srcs = ["test_log_playback.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/controller:log_playback",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
        "//nuplan/planning/simulation/trajectory:abstract_trajectory",
    ],
)

py_test(
    name = "test_perfect_tracking",
    size = "medium",
    srcs = ["test_perfect_tracking.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
        "//nuplan/planning/simulation/controller:perfect_tracking",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
        "//nuplan/planning/simulation/trajectory:interpolated_trajectory",
    ],
)

py_test(
    name = "test_utils",
    size = "small",
    srcs = ["test_utils.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/simulation/controller:utils",
    ],
)
