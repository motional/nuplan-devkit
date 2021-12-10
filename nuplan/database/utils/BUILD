load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "geometry",
    srcs = ["geometry.py"],
)

py_library(
    name = "measure",
    srcs = ["measure.py"],
    deps = [
        "//nuplan/database/utils:geometry",
        "//nuplan/database/utils/boxes:box3d",
    ],
)

py_library(
    name = "plot",
    srcs = ["plot.py"],
)
