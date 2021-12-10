load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_resnet",
    size = "medium",
    srcs = ["test_resnet.py"],
    deps = [
        "//nuplan/planning/training/modeling/nets:resnet",
    ],
)
