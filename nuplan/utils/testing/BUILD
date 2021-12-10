load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "instances",
    srcs = ["instances.py"],
)

py_binary(
    name = "plugin",
    srcs = ["plugin.py"],
    deps = [
    ],
)

py_library(
    name = "nuplan_test",
    srcs = ["nuplan_test.py"],
    deps = [
        ":instances",
        ":plugin",
    ],
)
