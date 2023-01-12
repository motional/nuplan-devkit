load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "scene_converter",
    srcs = ["scene_converter.py"],
    deps = [
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/training/modeling:types",
    ],
)

py_library(
    name = "scenario_scene_converter",
    srcs = ["scenario_scene_converter.py"],
    deps = [
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/callback:serialization_callback",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
        "//nuplan/planning/simulation/trajectory:interpolated_trajectory",
        "//nuplan/planning/training/callbacks/utils:scene_converter",
        "//nuplan/planning/training/modeling:types",
    ],
)

py_library(
    name = "visualization_utils",
    srcs = ["visualization_utils.py"],
    deps = [
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/planning/training/preprocessing/features:agents",
        "//nuplan/planning/training/preprocessing/features:generic_agents",
        "//nuplan/planning/training/preprocessing/features:raster",
        "//nuplan/planning/training/preprocessing/features:trajectory",
        "//nuplan/planning/training/preprocessing/features:vector_map",
        "//nuplan/planning/training/preprocessing/features:vector_set_map",
        requirement("opencv-python"),
    ],
)
