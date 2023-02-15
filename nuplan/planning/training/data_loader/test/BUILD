load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")
load("@pip_torch_deps//:requirements.bzl", requirement_torch = "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "skeleton_test_dataloader",
    srcs = ["skeleton_test_dataloader.py"],
    deps = [
        "//nuplan/planning/scenario_builder:scenario_filter",
        "//nuplan/planning/scenario_builder/cache:cached_scenario",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/scenario_builder/nuplan_db/test:nuplan_scenario_test_utils",
        "//nuplan/planning/training/data_augmentation:kinematic_agent_augmentation",
        "//nuplan/planning/training/data_loader:datamodule",
        "//nuplan/planning/training/data_loader:distributed_sampler_wrapper",
        "//nuplan/planning/training/data_loader:log_splitter",
        "//nuplan/planning/training/preprocessing:feature_preprocessor",
        "//nuplan/planning/training/preprocessing/feature_builders:agents_feature_builder",
        "//nuplan/planning/training/preprocessing/feature_builders:raster_feature_builder",
        "//nuplan/planning/training/preprocessing/feature_builders:vector_map_feature_builder",
        "//nuplan/planning/training/preprocessing/features:raster",
        "//nuplan/planning/training/preprocessing/features:trajectory",
        "//nuplan/planning/training/preprocessing/target_builders:ego_trajectory_target_builder",
        "//nuplan/planning/training/preprocessing/test:dummy_vectormap_builder",
        "//nuplan/planning/utils/multithreading:worker_pool",
        requirement("ray"),
        requirement_torch("pytorch-lightning"),
    ],
)

py_test(
    name = "test_dataloader_ray",
    size = "small",
    srcs = ["test_dataloader_ray.py"],
    tags = ["integration"],
    deps = [
        ":skeleton_test_dataloader",
        "//nuplan/planning/training/experiments:cache_metadata_entry",
        "//nuplan/planning/utils/multithreading:worker_pool",
        "//nuplan/planning/utils/multithreading:worker_ray",
    ],
)

py_test(
    name = "test_dataloader_sequential",
    size = "small",
    srcs = ["test_dataloader_sequential.py"],
    tags = ["integration"],
    deps = [
        ":skeleton_test_dataloader",
        "//nuplan/planning/training/experiments:cache_metadata_entry",
        "//nuplan/planning/utils/multithreading:worker_pool",
        "//nuplan/planning/utils/multithreading:worker_sequential",
    ],
)

py_test(
    name = "test_distributed_sampler_wrapper",
    size = "small",
    srcs = ["test_distributed_sampler_wrapper.py"],
    deps = [
        "//nuplan/planning/training/data_loader:distributed_sampler_wrapper",
        requirement_torch("torch"),
    ],
)

py_test(
    name = "test_distributed_weighted_sampler_init",
    size = "small",
    srcs = ["test_distributed_weighted_sampler_init.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/planning/scenario_builder/cache:cached_scenario",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_utils",
        "//nuplan/planning/training/data_loader:datamodule",
        "//nuplan/planning/training/data_loader:distributed_sampler_wrapper",
        "//nuplan/planning/training/data_loader:scenario_dataset",
        "//nuplan/planning/training/experiments:cache_metadata_entry",
        "//nuplan/planning/utils/multithreading:worker_utils",
        requirement("omegaconf"),
        requirement_torch("torch"),
        requirement_torch("pytorch-lightning"),
    ],
)
