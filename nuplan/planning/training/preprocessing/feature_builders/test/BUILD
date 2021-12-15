load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_agents_feature_builder",
    size = "medium",
    srcs = ["test_agents_feature_builder.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
        "//nuplan/planning/training/preprocessing/feature_builders:agents_feature_builder",
        "//nuplan/planning/training/preprocessing/features:agents",
    ],
)

py_test(
    name = "test_vector_map_feature_builder",
    size = "medium",
    srcs = ["test_vector_map_feature_builder.py"],
    deps = [
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/training/preprocessing/feature_builders:vector_map_feature_builder",
        "//nuplan/planning/training/preprocessing/features:vector_map",
    ],
)
