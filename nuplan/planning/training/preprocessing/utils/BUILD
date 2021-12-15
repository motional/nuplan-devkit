load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "agents_preprocessing",
    srcs = ["agents_preprocessing.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/database/utils/boxes:box3d",
        "//nuplan/planning/metrics/utils:state_extractors",
        "//nuplan/planning/training/preprocessing/features:abstract_model_feature",
        "//nuplan/planning/training/preprocessing/features:trajectory_utils",
    ],
)

py_library(
    name = "feature_cache",
    srcs = ["feature_cache.py"],
    deps = [
        "//nuplan/planning/training/preprocessing/feature_builders:abstract_feature_builder",
    ],
)

py_library(
    name = "utils_cache",
    srcs = ["utils_cache.py"],
    deps = [
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/training/preprocessing/feature_builders:abstract_feature_builder",
        "//nuplan/planning/training/preprocessing/target_builders:abstract_target_builder",
        "//nuplan/planning/training/preprocessing/utils:feature_cache",
    ],
)

py_library(
    name = "utils_scene",
    srcs = ["utils_scene.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/training/preprocessing/features:trajectory",
    ],
)
