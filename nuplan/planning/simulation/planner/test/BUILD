load("@rules_python//python:defs.bzl", "py_test")

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
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
    ],
)
