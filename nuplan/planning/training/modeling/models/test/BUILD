load("@rules_python//python:defs.bzl", "py_test")
load("@pip_torch_deps//:requirements.bzl", requirement_torch = "requirement")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_lanegcn",
    size = "medium",
    srcs = ["test_lanegcn.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/planning/simulation/trajectory:trajectory_sampling",
        "//nuplan/planning/training/modeling:types",
        "//nuplan/planning/training/modeling/models:lanegcn_model",
        "//nuplan/planning/training/modeling/models:lanegcn_utils",
        "//nuplan/planning/training/preprocessing/features:agents",
        "//nuplan/planning/training/preprocessing/features:trajectory",
        "//nuplan/planning/training/preprocessing/features:vector_map",
        requirement_torch("torch"),
    ],
)

py_test(
    name = "test_simple_vector_map_model",
    size = "medium",
    srcs = ["test_simple_vector_map_model.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/planning/simulation/trajectory:trajectory_sampling",
        "//nuplan/planning/training/modeling:types",
        "//nuplan/planning/training/modeling/models:simple_vector_map_model",
        "//nuplan/planning/training/preprocessing/features:agents",
        "//nuplan/planning/training/preprocessing/features:trajectory",
        "//nuplan/planning/training/preprocessing/features:vector_map",
        requirement_torch("torch"),
    ],
)

py_test(
    name = "test_urban_driver_open_loop_model",
    size = "small",
    srcs = ["test_urban_driver_open_loop_model.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/planning/simulation/trajectory:trajectory_sampling",
        "//nuplan/planning/training/modeling:types",
        "//nuplan/planning/training/modeling/models:urban_driver_open_loop_model",
        "//nuplan/planning/training/modeling/models:urban_driver_open_loop_model_utils",
        "//nuplan/planning/training/preprocessing/features:generic_agents",
        "//nuplan/planning/training/preprocessing/features:trajectory",
        "//nuplan/planning/training/preprocessing/features:vector_set_map",
        requirement_torch("torch"),
    ],
)
