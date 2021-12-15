load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "dummy_vectormap_builder",
    srcs = ["dummy_vectormap_builder.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/training/preprocessing/feature_builders:abstract_feature_builder",
        "//nuplan/planning/training/preprocessing/features:abstract_model_feature",
    ],
)

py_test(
    name = "test_collate",
    size = "medium",
    srcs = ["test_collate.py"],
    deps = [
        "//nuplan/planning/training/modeling:types",
        "//nuplan/planning/training/preprocessing:feature_collate",
        "//nuplan/planning/training/preprocessing/features:raster",
        "//nuplan/planning/training/preprocessing/features:trajectory",
        "//nuplan/planning/training/preprocessing/test:dummy_vectormap_builder",
    ],
)

py_test(
    name = "test_collate_dataloader",
    size = "medium",
    srcs = ["test_collate_dataloader.py"],
    deps = [
        "//nuplan/planning/scenario_builder:scenario_filter",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/training/data_loader:scenario_dataset",
        "//nuplan/planning/training/preprocessing:feature_caching_preprocessor",
        "//nuplan/planning/training/preprocessing:feature_collate",
        "//nuplan/planning/training/preprocessing/feature_builders:raster_feature_builder",
        "//nuplan/planning/training/preprocessing/feature_builders:vector_map_feature_builder",
        "//nuplan/planning/training/preprocessing/features:vector_map",
        "//nuplan/planning/training/preprocessing/target_builders:ego_trajectory_target_builder",
        "//nuplan/planning/utils/multithreading:worker_sequential",
    ],
)

py_test(
    name = "test_feature_caching_preprocessor",
    size = "medium",
    srcs = ["test_feature_caching_preprocessor.py"],
    deps = [
        "//nuplan/planning/scenario_builder:scenario_filter",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/training/preprocessing:feature_caching_preprocessor",
        "//nuplan/planning/training/preprocessing/feature_builders:raster_feature_builder",
        "//nuplan/planning/training/preprocessing/feature_builders:vector_map_feature_builder",
        "//nuplan/planning/training/preprocessing/target_builders:ego_trajectory_target_builder",
        "//nuplan/planning/utils/multithreading:worker_sequential",
    ],
)

py_test(
    name = "test_utils_cache",
    size = "medium",
    srcs = ["test_utils_cache.py"],
    deps = [
        "//nuplan/planning/training/preprocessing/features:abstract_model_feature",
        "//nuplan/planning/training/preprocessing/features:raster",
        "//nuplan/planning/training/preprocessing/features:vector_map",
        "//nuplan/planning/training/preprocessing/utils:feature_cache",
    ],
)
