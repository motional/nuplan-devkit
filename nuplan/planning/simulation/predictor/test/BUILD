load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "mock_abstract_predictor",
    srcs = ["mock_abstract_predictor.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history_buffer",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/predictor:abstract_predictor",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
    ],
)

py_test(
    name = "test_abstract_predictor",
    size = "small",
    srcs = ["test_abstract_predictor.py"],
    deps = [
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/predictor:predictor_report",
        "//nuplan/planning/simulation/predictor/test:mock_abstract_predictor",
    ],
)

py_test(
    name = "test_log_future_predictor",
    size = "small",
    srcs = ["test_log_future_predictor.py"],
    deps = [
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
        "//nuplan/planning/simulation/predictor:log_future_predictor",
        "//nuplan/planning/simulation/predictor:predictor_report",
        "//nuplan/planning/simulation/predictor/test:mock_abstract_predictor",
        "//nuplan/planning/simulation/trajectory:trajectory_sampling",
    ],
)
