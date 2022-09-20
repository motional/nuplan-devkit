load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")
load("@pip_torch_deps//:requirements.bzl", requirement_torch = "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "abstract_model_feature",
    srcs = ["abstract_model_feature.py"],
    deps = [
        requirement("numpy"),
        requirement_torch("torch"),
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
    name = "generic_agents",
    srcs = ["generic_agents.py"],
    deps = [
        "//nuplan/common/actor_state:tracked_objects_types",
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
        requirement_torch("torchvision"),
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
        requirement("opencv-python"),
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
        "//nuplan/planning/training/preprocessing/feature_builders:vector_builder_utils",
        "//nuplan/planning/training/preprocessing/features:abstract_model_feature",
        "//nuplan/planning/training/preprocessing/features:vector_utils",
        requirement("pyquaternion"),
    ],
)

py_library(
    name = "vector_set_map",
    srcs = ["vector_set_map.py"],
    deps = [
        "//nuplan/planning/script/builders/utils:utils_type",
        "//nuplan/planning/training/preprocessing/feature_builders:vector_builder_utils",
        "//nuplan/planning/training/preprocessing/features:abstract_model_feature",
        "//nuplan/planning/training/preprocessing/features:vector_utils",
        requirement("pyquaternion"),
    ],
)

py_library(
    name = "vector_utils",
    srcs = ["vector_utils.py"],
    deps = [
        "//nuplan/planning/script/builders/utils:utils_type",
        "//nuplan/planning/training/preprocessing/features:abstract_model_feature",
    ],
)
