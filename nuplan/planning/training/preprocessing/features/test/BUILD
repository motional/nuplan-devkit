load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_agents",
    size = "medium",
    srcs = ["test_agents.py"],
    deps = [
        "//nuplan/planning/training/preprocessing/features:agents",
    ],
)

py_test(
    name = "test_raster_building",
    size = "medium",
    srcs = ["test_raster_building.py"],
    deps = [
        "//nuplan/planning/scenario_builder:scenario_filter",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/training/preprocessing/features:raster_utils",
        "//nuplan/planning/utils/multithreading:worker_sequential",
    ],
)

py_test(
    name = "test_trajectory",
    size = "medium",
    srcs = ["test_trajectory.py"],
    deps = [
        "//nuplan/planning/training/preprocessing/features:trajectory",
    ],
)
