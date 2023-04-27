load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")
load("@pip_torch_deps//:requirements.bzl", requirement_torch = "requirement")

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

py_library(
    name = "skeleton_test_serialization_callback",
    srcs = ["skeleton_test_serialization_callback.py"],
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
        "//nuplan/planning/simulation/simulation_time_controller:abstract_simulation_time_controller",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
        "//nuplan/planning/simulation/trajectory:interpolated_trajectory",
        requirement("hypothesis"),
        requirement("msgpack"),
    ],
)

py_test(
    name = "test_serialization_callback_json",
    size = "small",
    srcs = ["test_serialization_callback_json.py"],
    deps = [
        "//nuplan/planning/simulation/callback/test:skeleton_test_serialization_callback",
    ],
)

py_test(
    name = "test_serialization_callback_msgpack",
    size = "small",
    srcs = ["test_serialization_callback_msgpack.py"],
    deps = [
        "//nuplan/planning/simulation/callback/test:skeleton_test_serialization_callback",
    ],
)

py_test(
    name = "test_serialization_callback_pickle",
    size = "small",
    srcs = ["test_serialization_callback_pickle.py"],
    deps = [
        "//nuplan/planning/simulation/callback/test:skeleton_test_serialization_callback",
    ],
)

py_test(
    name = "test_simulation_log_callback",
    size = "medium",
    srcs = ["test_simulation_log_callback.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
        "//nuplan/planning/simulation:simulation_log",
        "//nuplan/planning/simulation:simulation_setup",
        "//nuplan/planning/simulation/callback:serialization_callback",
        "//nuplan/planning/simulation/callback:simulation_log_callback",
        "//nuplan/planning/simulation/controller:abstract_controller",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/observation:abstract_observation",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/planner:simple_planner",
        "//nuplan/planning/simulation/simulation_time_controller:abstract_simulation_time_controller",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
        "//nuplan/planning/simulation/trajectory:interpolated_trajectory",
    ],
)

py_test(
    name = "test_simulation_log",
    size = "small",
    srcs = ["test_simulation_log.py"],
    deps = [
        "//nuplan/planning/simulation:simulation_log",
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
        "//nuplan/planning/simulation/trajectory:abstract_trajectory",
        requirement_torch("torch"),
        requirement_torch("tensorboard"),
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
