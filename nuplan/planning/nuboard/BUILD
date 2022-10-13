load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "nuboard",
    srcs = ["nuboard.py"],
    deps = [
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/planning/nuboard/base:experiment_file_data",
        "//nuplan/planning/nuboard/tabs:cloud_tab",
        "//nuplan/planning/nuboard/tabs:configuration_tab",
        "//nuplan/planning/nuboard/tabs:histogram_tab",
        "//nuplan/planning/nuboard/tabs:overview_tab",
        "//nuplan/planning/nuboard/tabs:scenario_tab",
        "//nuplan/planning/nuboard/utils",
        "//nuplan/planning/scenario_builder:abstract_scenario_builder",
        "//nuplan/planning/training/callbacks:profile_callback",
        requirement("jinja2"),
        requirement("bokeh"),
    ],
)

py_library(
    name = "style",
    srcs = ["style.py"],
    deps = [
        "//nuplan/common/maps:abstract_map",
    ],
)
