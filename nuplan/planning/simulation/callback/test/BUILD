load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_metric_callback",
    size = "medium",
    srcs = ["test_metric_callback.py"],
    deps = [
        "//nuplan/planning/metrics:metric_engine",
        "//nuplan/planning/simulation/callback:metric_callback",
        "//nuplan/planning/simulation/planner:abstract_planner",
    ],
)

py_test(
    name = "test_serialization_callback",
    size = "medium",
    srcs = ["test_serialization_callback.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
        "//nuplan/planning/simulation:simulation_setup",
        "//nuplan/planning/simulation/callback:serialization_callback",
        "//nuplan/planning/simulation/controller:abstract_controller",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/observation:abstract_observation",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/simulation_manager:abstract_simulation_manager",
        "//nuplan/planning/simulation/simulation_manager:simulation_iteration",
        "//nuplan/planning/simulation/trajectory:interpolated",
    ],
)

py_test(
    name = "test_timing_callback",
    size = "medium",
    srcs = ["test_timing_callback.py"],
    deps = [
        "//nuplan/planning/simulation:simulation_setup",
        "//nuplan/planning/simulation/callback:timing_callback",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/trajectory",
    ],
)

py_test(
    name = "test_visualization_callback",
    size = "medium",
    srcs = ["test_visualization_callback.py"],
    deps = [
        "//nuplan/planning/simulation:simulation_setup",
        "//nuplan/planning/simulation/callback:visualization_callback",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/visualization:abstract_visualization",
    ],
)
