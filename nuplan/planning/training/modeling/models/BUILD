load("@rules_python//python:defs.bzl", "py_library")
load("@pip_torch_deps//:requirements.bzl", requirement_torch = "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "lanegcn_utils",
    srcs = ["lanegcn_utils.py"],
    deps = [
    ],
)

py_library(
    name = "lanegcn_model",
    srcs = ["lanegcn_model.py"],
    deps = [
        "//nuplan/planning/training/modeling:torch_module_wrapper",
        "//nuplan/planning/training/modeling:types",
        "//nuplan/planning/training/modeling/models:lanegcn_utils",
        "//nuplan/planning/training/preprocessing/feature_builders:agents_feature_builder",
        "//nuplan/planning/training/preprocessing/feature_builders:vector_map_feature_builder",
        "//nuplan/planning/training/preprocessing/features:agents",
        "//nuplan/planning/training/preprocessing/features:trajectory",
        "//nuplan/planning/training/preprocessing/features:vector_map",
        "//nuplan/planning/training/preprocessing/target_builders:ego_trajectory_target_builder",
        requirement_torch("torch"),
    ],
)

py_library(
    name = "raster_model",
    srcs = ["raster_model.py"],
    deps = [
        "//nuplan/planning/training/modeling:torch_module_wrapper",
        "//nuplan/planning/training/modeling:types",
        "//nuplan/planning/training/preprocessing/feature_builders:abstract_feature_builder",
        "//nuplan/planning/training/preprocessing/feature_builders:raster_feature_builder",
        "//nuplan/planning/training/preprocessing/features:raster",
        "//nuplan/planning/training/preprocessing/features:trajectory",
        "//nuplan/planning/training/preprocessing/target_builders:abstract_target_builder",
        "//nuplan/planning/training/preprocessing/target_builders:ego_trajectory_target_builder",
        requirement_torch("timm"),
    ],
)

py_library(
    name = "simple_vector_map_model",
    srcs = ["simple_vector_map_model.py"],
    deps = [
        "//nuplan/planning/training/modeling:scriptable_torch_module_wrapper",
        "//nuplan/planning/training/modeling:types",
        "//nuplan/planning/training/preprocessing/feature_builders:agents_feature_builder",
        "//nuplan/planning/training/preprocessing/feature_builders:vector_map_feature_builder",
        "//nuplan/planning/training/preprocessing/features:agents",
        "//nuplan/planning/training/preprocessing/features:trajectory",
        "//nuplan/planning/training/preprocessing/features:vector_map",
        "//nuplan/planning/training/preprocessing/target_builders:ego_trajectory_target_builder",
        requirement_torch("torch"),
    ],
)

py_library(
    name = "urban_driver_open_loop_model",
    srcs = ["urban_driver_open_loop_model.py"],
    deps = [
        "//nuplan/planning/simulation/trajectory:trajectory_sampling",
        "//nuplan/planning/training/modeling:torch_module_wrapper",
        "//nuplan/planning/training/modeling:types",
        "//nuplan/planning/training/modeling/models:urban_driver_open_loop_model_utils",
        "//nuplan/planning/training/preprocessing/feature_builders:generic_agents_feature_builder",
        "//nuplan/planning/training/preprocessing/feature_builders:vector_set_map_feature_builder",
        "//nuplan/planning/training/preprocessing/features:generic_agents",
        "//nuplan/planning/training/preprocessing/features:trajectory",
        "//nuplan/planning/training/preprocessing/features:vector_set_map",
        "//nuplan/planning/training/preprocessing/target_builders:ego_trajectory_target_builder",
        requirement_torch("torch"),
    ],
)

py_library(
    name = "urban_driver_open_loop_model_utils",
    srcs = ["urban_driver_open_loop_model_utils.py"],
    deps = [
        requirement_torch("torch"),
    ],
)
