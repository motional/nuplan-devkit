load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = ["//nuplan/planning/nuboard/base:data_class"],
)

py_library(
    name = "nuboard_cloud_utils",
    srcs = ["nuboard_cloud_utils.py"],
    deps = [
        "//nuplan/common/utils:s3_utils",
        "//nuplan/planning/nuboard/base:data_class",
    ],
)
