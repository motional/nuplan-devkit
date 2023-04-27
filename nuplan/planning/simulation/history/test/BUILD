load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_simulation_history",
    size = "small",
    srcs = ["test_simulation_history.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
    ],
)

py_test(
    name = "test_simulation_history_buffer",
    size = "small",
    srcs = ["test_simulation_history_buffer.py"],
    deps = [
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history_buffer",
        "//nuplan/planning/simulation/observation:lidar_pc",
        "//nuplan/planning/simulation/observation:tracks_observation",
    ],
)
