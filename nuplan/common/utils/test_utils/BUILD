load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "instances",
    srcs = ["instances.py"],
)

py_library(
    name = "patch",
    srcs = ["patch.py"],
    deps = [
        ":function_validation",
    ],
)

py_library(
    name = "function_validation",
    srcs = ["function_validation.py"],
)

py_library(
    name = "interface_validation",
    srcs = ["interface_validation.py"],
    deps = [
        ":function_validation",
    ],
)

py_binary(
    name = "plugin",
    srcs = ["plugin.py"],
)

py_library(
    name = "nuplan_test",
    srcs = ["nuplan_test.py"],
    deps = [
        "//nuplan/common/utils/test_utils:instances",
        "//nuplan/common/utils/test_utils:plugin",
    ],
)

py_library(
    name = "mock_s3_utils",
    testonly = True,
    srcs = ["mock_s3_utils.py"],
    deps = [
        requirement("moto"),
        requirement("aiofiles"),
        requirement("aiohttp"),
        requirement("aiobotocore"),
        requirement("botocore"),
    ],
)
