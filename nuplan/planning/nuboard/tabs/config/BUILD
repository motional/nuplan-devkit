load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "cloud_tab_config",
    srcs = ["cloud_tab_config.py"],
    deps = [
        requirement("bokeh"),
    ],
)

py_library(
    name = "scenario_tab_config",
    srcs = ["scenario_tab_config.py"],
)

py_library(
    name = "overview_tab_config",
    srcs = ["overview_tab_config.py"],
)

py_library(
    name = "histogram_tab_config",
    srcs = ["histogram_tab_config.py"],
    deps = [
        "//nuplan/planning/nuboard:style",
        requirement("bokeh"),
        requirement("numpy"),
    ],
)
