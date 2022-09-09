load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_test(
    name = "test_nuboard_utils",
    size = "medium",
    srcs = ["test_nuboard_utils.py"],
    deps = [
        "//nuplan/planning/nuboard/base:data_class",
        "//nuplan/planning/nuboard/utils",
    ],
)

py_test(
    name = "test_nuboard_cloud_utils",
    size = "small",
    srcs = ["test_nuboard_cloud_utils.py"],
    deps = [
        "//nuplan/planning/nuboard/base:data_class",
        "//nuplan/planning/nuboard/utils:nuboard_cloud_utils",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
        "//nuplan/planning/simulation:simulation_log",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/planner:simple_planner",
        "//nuplan/planning/simulation/simulation_time_controller:abstract_simulation_time_controller",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
        "//nuplan/planning/simulation/trajectory:interpolated_trajectory",
    ],
)
