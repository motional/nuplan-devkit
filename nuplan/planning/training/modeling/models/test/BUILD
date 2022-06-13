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
