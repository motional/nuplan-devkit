load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "label",
    srcs = ["label.py"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = ["//nuplan/database/utils/label"],
)
