load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@pip_torch_deps//:requirements.bzl", requirement_torch = "requirement")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "patch_test_methods",
    testonly = True,
    srcs = ["patch_test_methods.py"],
    deps = [],
)

py_test(
    name = "test_patch",
    size = "small",
    srcs = ["test_patch.py"],
    deps = [
        ":patch_test_methods",
        "//nuplan/common/utils/test_utils:patch",
    ],
)

py_test(
    name = "test_interface_validation",
    size = "small",
    srcs = ["test_interface_validation.py"],
    deps = [
        "//nuplan/common/utils/test_utils:interface_validation",
    ],
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
    name = "test_function_validation",
    size = "small",
    srcs = ["test_function_validation.py"],
    deps = [
        "//nuplan/common/utils/test_utils:function_validation",
        requirement("numpy"),
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
    tags = ["manual"],
    deps = [
        "//nuplan/common/utils:s3_utils",
        "//nuplan/common/utils/test_utils:mock_s3_utils",
    ],
)

py_test(
    name = "test_distributed_scenario_filter",
    size = "small",
    srcs = ["test_distributed_scenario_filter.py"],
    deps = [
        "//nuplan/common/utils:distributed_scenario_filter",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/scenario_builder:abstract_scenario_builder",
        "//nuplan/planning/scenario_builder:scenario_filter",
        "//nuplan/planning/utils/multithreading:worker_pool",
        requirement("pandas"),
        requirement("omegaconf"),
    ],
)

py_test(
    name = "test_io_utils",
    size = "small",
    srcs = ["test_io_utils.py"],
    deps = [
        "//nuplan/common/utils:io_utils",
        "//nuplan/common/utils/test_utils:patch",
    ],
)
