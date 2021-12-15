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
    ],
)

py_library(
    name = "histogram_tab",
    srcs = ["histogram_tab.py"],
    deps = [
        "//nuplan/planning/metrics:metric_file",
        "//nuplan/planning/nuboard:style",
        "//nuplan/planning/nuboard/base:base_tab",
        "//nuplan/planning/nuboard/base:data_class",
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
        "//nuplan/planning/nuboard/base:simulation_tile",
        "//nuplan/planning/scenario_builder:abstract_scenario_builder",
    ],
)
