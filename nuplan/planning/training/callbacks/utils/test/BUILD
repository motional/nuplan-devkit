load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_scenario_scoring_callback",
    size = "small",
    srcs = ["test_scenario_scoring_callback.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
        "//nuplan/planning/training/callbacks:scenario_scoring_callback",
        "//nuplan/planning/training/callbacks/utils:scenario_scene_converter",
        "//nuplan/planning/training/data_loader:scenario_dataset",
        "//nuplan/planning/training/modeling:types",
        "//nuplan/planning/training/preprocessing/test:dummy_vectormap_builder",
    ],
)

py_test(
    name = "test_visualization_utils",
    size = "small",
    srcs = ["test_visualization_utils.py"],
    deps = [
        "//nuplan/planning/training/callbacks/utils:visualization_utils",
        "//nuplan/planning/training/preprocessing/features:agents",
        "//nuplan/planning/training/preprocessing/features:raster",
        "//nuplan/planning/training/preprocessing/features:trajectory",
        "//nuplan/planning/training/preprocessing/features:vector_map",
    ],
)
