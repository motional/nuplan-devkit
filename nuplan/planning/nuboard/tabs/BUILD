load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "configuration_tab",
    srcs = ["configuration_tab.py"],
    deps = [
        "//nuplan/planning/nuboard:style",
        "//nuplan/planning/nuboard/base:base_tab",
        "//nuplan/planning/nuboard/base:data_class",
        "//nuplan/planning/nuboard/base:experiment_file_data",
    ],
)

py_library(
    name = "histogram_tab",
    srcs = ["histogram_tab.py"],
    deps = [
        "//nuplan/planning/metrics:metric_file",
        "//nuplan/planning/nuboard/base:base_tab",
        "//nuplan/planning/nuboard/base:data_class",
        "//nuplan/planning/nuboard/base:experiment_file_data",
        "//nuplan/planning/nuboard/tabs/config:histogram_tab_config",
        "//nuplan/planning/nuboard/tabs/js_code:histogram_tab_js_code",
    ],
)

py_library(
    name = "overview_tab",
    srcs = ["overview_tab.py"],
    deps = [
        "//nuplan/planning/metrics:metric_file",
        "//nuplan/planning/nuboard:style",
        "//nuplan/planning/nuboard/base:base_tab",
        "//nuplan/planning/nuboard/base:data_class",
        "//nuplan/planning/nuboard/base:experiment_file_data",
        "//nuplan/planning/nuboard/tabs/config:overview_tab_config",
    ],
)

py_library(
    name = "scenario_tab",
    srcs = ["scenario_tab.py"],
    deps = [
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/planning/nuboard:style",
        "//nuplan/planning/nuboard/base:base_tab",
        "//nuplan/planning/nuboard/base:data_class",
        "//nuplan/planning/nuboard/base:experiment_file_data",
        "//nuplan/planning/nuboard/base:plot_data",
        "//nuplan/planning/nuboard/base:simulation_tile",
        "//nuplan/planning/nuboard/tabs/config:scenario_tab_config",
        "//nuplan/planning/nuboard/tabs/js_code:scenario_tab_js_code",
        "//nuplan/planning/scenario_builder:abstract_scenario_builder",
    ],
)

py_library(
    name = "cloud_tab",
    srcs = ["cloud_tab.py"],
    deps = [
        ":configuration_tab",
        "//nuplan/common/utils:s3_utils",
        "//nuplan/planning/nuboard:style",
        "//nuplan/planning/nuboard/tabs/config:cloud_tab_config",
        "//nuplan/planning/nuboard/tabs/js_code:cloud_tab_js_code",
        "//nuplan/planning/nuboard/utils:nuboard_cloud_utils",
    ],
)
