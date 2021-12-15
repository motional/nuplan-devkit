load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "abstract_model_feature",
    srcs = ["abstract_model_feature.py"],
    deps = [
    ],
)

py_library(
    name = "agents",
    srcs = ["agents.py"],
    deps = [
        "//nuplan/planning/training/preprocessing/features:abstract_model_feature",
    ],
)

py_library(
    name = "agents_trajectories",
    srcs = ["agents_trajectories.py"],
    deps = [
        "//nuplan/planning/training/preprocessing/features:abstract_model_feature",
    ],
)

py_library(
    name = "raster",
    srcs = ["raster.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/planning/script/builders/utils:utils_type",
        "//nuplan/planning/training/preprocessing/features:abstract_model_feature",
    ],
)

py_library(
    name = "raster_utils",
    srcs = ["raster_utils.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/planning/simulation/observation:observation_type",
    ],
)

py_library(
    name = "trajectories",
    srcs = ["trajectories.py"],
    deps = [
        "//nuplan/planning/training/preprocessing/features:abstract_model_feature",
        "//nuplan/planning/training/preprocessing/features:trajectory",
    ],
)

py_library(
    name = "trajectory",
    srcs = ["trajectory.py"],
    deps = [
        "//nuplan/planning/script/builders/utils:utils_type",
        "//nuplan/planning/training/preprocessing/feature_builders:abstract_feature_builder",
        "//nuplan/planning/training/preprocessing/features:abstract_model_feature",
    ],
)

py_library(
    name = "trajectory_utils",
    srcs = ["trajectory_utils.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
    ],
)

py_library(
    name = "vector_map",
    srcs = ["vector_map.py"],
    deps = [
        "//nuplan/planning/script/builders/utils:utils_type",
        "//nuplan/planning/training/preprocessing/features:abstract_model_feature",
    ],
)
