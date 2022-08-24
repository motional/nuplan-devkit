load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_evalai_interface",
    size = "small",
    srcs = ["test_evalai_interface.py"],
    deps = [
        "//nuplan/submission/evalai:evalai_interface",
    ],
)
