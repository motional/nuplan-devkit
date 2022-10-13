load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//nuplan/planning/metrics:metric_dataframe",
        "//nuplan/planning/nuboard/base:data_class",
        requirement("pandas"),
    ],
)

py_library(
    name = "nuboard_cloud_utils",
    srcs = ["nuboard_cloud_utils.py"],
    deps = [
        "//nuplan/common/utils:s3_utils",
        "//nuplan/planning/nuboard/base:data_class",
        requirement("numpy"),
        requirement("boto3"),
    ],
)

py_library(
    name = "nuboard_histogram_utils",
    srcs = ["nuboard_histogram_utils.py"],
    deps = [
        "//nuplan/planning/metrics:metric_dataframe",
        "//nuplan/planning/nuboard/tabs/config:histogram_tab_config",
        requirement("numpy"),
        requirement("pandas"),
        requirement("bokeh"),
    ],
)
