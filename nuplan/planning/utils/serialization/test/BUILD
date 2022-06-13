load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_test(
    name = "test_scene_simple_trajectory",
    size = "small",
    srcs = ["test_scene_simple_trajectory.py"],
    deps = [
        "//nuplan/planning/utils/serialization:scene_simple_trajectory",
    ],
)
