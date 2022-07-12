load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_lanegcn",
    size = "medium",
    srcs = ["test_lanegcn.py"],
    deps = [
        "//nuplan/planning/training/modeling/models:lanegcn_utils",
    ],
)

py_test(
    name = "test_simple_vector_map_model",
    size = "medium",
    srcs = ["test_simple_vector_map_model.py"],
    deps = [
        "//nuplan/planning/simulation/trajectory:trajectory_sampling",
        "//nuplan/planning/training/modeling:types",
        "//nuplan/planning/training/modeling/models:simple_vector_map_model",
        "//nuplan/planning/training/preprocessing/features:agents",
        "//nuplan/planning/training/preprocessing/features:trajectory",
        "//nuplan/planning/training/preprocessing/features:vector_map",
    ],
)
