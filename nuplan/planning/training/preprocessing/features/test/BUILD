load("@rules_python//python:defs.bzl", "py_test")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

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
    name = "test_generic_agents",
    size = "medium",
    srcs = ["test_generic_agents.py"],
    deps = [
        "//nuplan/planning/training/preprocessing/features:generic_agents",
    ],
)

py_test(
    name = "test_raster_building",
    size = "medium",
    srcs = ["test_raster_building.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/planning/scenario_builder:scenario_filter",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/scenario_builder/nuplan_db/test:nuplan_scenario_test_utils",
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

py_test(
    name = "test_trajectories",
    size = "medium",
    srcs = ["test_trajectories.py"],
    deps = [
        "//nuplan/planning/training/preprocessing/features:trajectories",
        "//nuplan/planning/training/preprocessing/features:trajectory",
    ],
)

py_test(
    name = "test_vector_set_map",
    size = "medium",
    srcs = ["test_vector_set_map.py"],
    deps = [
        "//nuplan/planning/training/preprocessing/features:vector_set_map",
    ],
)

py_test(
    name = "test_vector_utils",
    size = "medium",
    srcs = ["test_vector_utils.py"],
    deps = [
        "//nuplan/planning/training/preprocessing/features:vector_utils",
        requirement("pyquaternion"),
    ],
)
