load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@pip_torch_deps//:requirements.bzl", requirement_torch = "requirement")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_test(
    name = "test_helpers",
    size = "small",
    srcs = ["test_helpers.py"],
    deps = [
        "//nuplan/common/utils:helpers",
    ],
)

py_test(
    name = "test_torch_math",
    size = "small",
    srcs = ["test_torch_math.py"],
    deps = [
        "//nuplan/common/utils:torch_math",
        requirement_torch("torch"),
        requirement("numpy"),
    ],
)

py_test(
    name = "test_file_backed_barrier",
    size = "small",
    srcs = ["test_file_backed_barrier.py"],
    deps = [
        "//nuplan/common/utils:file_backed_barrier",
    ],
)

py_test(
    name = "test_s3_utils",
    size = "small",
    srcs = ["test_s3_utils.py"],
    deps = [
        "//nuplan/common/utils:s3_utils",
    ],
)
