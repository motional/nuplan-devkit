load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_log",
    size = "medium",
    srcs = ["test_log.py"],
    deps = [
        "//nuplan/planning/script/builders:logging_builder",
    ],
)
