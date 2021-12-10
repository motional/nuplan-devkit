load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_test(
    name = "test_label",
    size = "small",
    srcs = ["test_label.py"],
    deps = ["//nuplan/database/utils/label"],
)

py_test(
    name = "test_utils",
    size = "small",
    srcs = ["test_utils.py"],
    deps = [
        "//nuplan/database/utils/label",
        "//nuplan/database/utils/label:utils",
    ],
)
