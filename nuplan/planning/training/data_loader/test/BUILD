load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_dataloader",
    size = "medium",
    srcs = ["test_dataloader.py"],
    deps = [
        "//nuplan/planning/scenario_builder:scenario_filter",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/training/data_loader:datamodule",
        "//nuplan/planning/training/data_loader:log_splitter",
        "//nuplan/planning/training/preprocessing:feature_caching_preprocessor",
        "//nuplan/planning/training/preprocessing/feature_builders:raster_feature_builder",
        "//nuplan/planning/training/preprocessing/feature_builders:vector_map_feature_builder",
        "//nuplan/planning/training/preprocessing/features:raster",
        "//nuplan/planning/training/preprocessing/features:trajectory",
        "//nuplan/planning/training/preprocessing/target_builders:ego_trajectory_target_builder",
        "//nuplan/planning/training/preprocessing/test:dummy_vectormap_builder",
        "//nuplan/planning/utils/multithreading:worker_pool",
        "//nuplan/planning/utils/multithreading:worker_ray",
        "//nuplan/planning/utils/multithreading:worker_sequential",
    ],
)
