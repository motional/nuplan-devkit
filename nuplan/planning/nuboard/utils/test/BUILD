load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_test(
    name = "test_nuboard_utils",
    size = "medium",
    srcs = ["test_nuboard_utils.py"],
    deps = [
        "//nuplan/planning/nuboard/base:data_class",
        "//nuplan/planning/nuboard/utils",
    ],
)
