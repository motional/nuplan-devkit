load("@rules_python//python:defs.bzl", "py_test")
load("@pip_torch_deps//:requirements.bzl", requirement_torch = "requirement")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_scenario_scoring_callback",
    size = "small",
    srcs = ["test_scenario_scoring_callback.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/scenario_builder/cache:cached_scenario",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
        "//nuplan/planning/training/callbacks:scenario_scoring_callback",
        "//nuplan/planning/training/callbacks/utils:scenario_scene_converter",
        "//nuplan/planning/training/data_augmentation:data_augmentation_util",
        "//nuplan/planning/training/data_loader:distributed_sampler_wrapper",
        "//nuplan/planning/training/data_loader:scenario_dataset",
        "//nuplan/planning/training/experiments:cache_metadata_entry",
        "//nuplan/planning/training/modeling:types",
        "//nuplan/planning/training/preprocessing/test:dummy_vectormap_builder",
        "//nuplan/planning/utils/multithreading:worker_utils",
        requirement_torch("torch"),
    ],
)

py_test(
    name = "test_visualization_utils",
    size = "small",
    srcs = ["test_visualization_utils.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/planning/training/callbacks/utils:visualization_utils",
        "//nuplan/planning/training/preprocessing/feature_builders:vector_builder_utils",
        "//nuplan/planning/training/preprocessing/features:agents",
        "//nuplan/planning/training/preprocessing/features:generic_agents",
        "//nuplan/planning/training/preprocessing/features:raster",
        "//nuplan/planning/training/preprocessing/features:trajectory",
        "//nuplan/planning/training/preprocessing/features:vector_map",
        "//nuplan/planning/training/preprocessing/features:vector_set_map",
    ],
)

py_test(
    name = "test_stepwise_augmentation_scheduler",
    size = "small",
    srcs = ["test_stepwise_augmentation_scheduler.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/planning/training/callbacks:stepwise_augmentation_scheduler",
        "//nuplan/planning/training/data_augmentation:abstract_data_augmentation",
        "//nuplan/planning/training/data_augmentation:data_augmentation_util",
        requirement_torch("pytorch-lightning"),
    ],
)
