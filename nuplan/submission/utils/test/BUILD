load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_aws_utils",
    size = "small",
    srcs = ["test_aws_utils.py"],
    deps = [
        "@nuplan_devkit//nuplan/submission/utils:aws_utils",
    ],
)

py_test(
    name = "test_utils",
    size = "small",
    srcs = ["test_utils.py"],
    deps = [
        "@nuplan_devkit//nuplan/submission/utils",
    ],
)
