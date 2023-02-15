load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")
load("@pip_torch_deps//:requirements.bzl", requirement_torch = "requirement")

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
    tags = ["integration"],
    deps = [
        "//nuplan/planning/scenario_builder/cache:cached_scenario",
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
    tags = ["integration"],
    deps = [
        "//nuplan/planning/scenario_builder:scenario_filter",
        "//nuplan/planning/scenario_builder/cache:cached_scenario",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/scenario_builder/nuplan_db/test:nuplan_scenario_test_utils",
        "//nuplan/planning/training/data_augmentation:kinematic_agent_augmentation",
        "//nuplan/planning/training/data_loader:scenario_dataset",
        "//nuplan/planning/training/experiments:cache_metadata_entry",
        "//nuplan/planning/training/preprocessing:feature_collate",
        "//nuplan/planning/training/preprocessing:feature_preprocessor",
        "//nuplan/planning/training/preprocessing/feature_builders:raster_feature_builder",
        "//nuplan/planning/training/preprocessing/feature_builders:vector_map_feature_builder",
        "//nuplan/planning/training/preprocessing/features:vector_map",
        "//nuplan/planning/training/preprocessing/target_builders:ego_trajectory_target_builder",
        "//nuplan/planning/utils/multithreading:worker_pool",
        "//nuplan/planning/utils/multithreading:worker_sequential",
        requirement_torch("torch"),
    ],
)

py_test(
    name = "test_feature_preprocessor",
    size = "medium",
    srcs = ["test_feature_preprocessor.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/planning/scenario_builder:scenario_filter",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/scenario_builder/nuplan_db/test:nuplan_scenario_test_utils",
        "//nuplan/planning/training/experiments:cache_metadata_entry",
        "//nuplan/planning/training/preprocessing:feature_preprocessor",
        "//nuplan/planning/training/preprocessing/feature_builders:raster_feature_builder",
        "//nuplan/planning/training/preprocessing/feature_builders:vector_map_feature_builder",
        "//nuplan/planning/training/preprocessing/target_builders:ego_trajectory_target_builder",
        "//nuplan/planning/utils/multithreading:worker_sequential",
        "//nuplan/planning/utils/multithreading:worker_utils",
        requirement("mock"),
    ],
)

py_test(
    name = "test_utils_cache",
    size = "medium",
    srcs = ["test_utils_cache.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/database/common/blob_store/test:mock_s3_store",
        "//nuplan/planning/training/preprocessing/features:abstract_model_feature",
        "//nuplan/planning/training/preprocessing/features:raster",
        "//nuplan/planning/training/preprocessing/features:vector_map",
        "//nuplan/planning/training/preprocessing/utils:feature_cache",
    ],
)
