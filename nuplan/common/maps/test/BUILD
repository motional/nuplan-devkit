load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_test(
    name = "test_map_manager",
    size = "small",
    srcs = ["test_map_manager.py"],
    deps = [
        "//nuplan/common/maps:map_manager",
    ],
)
