load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_test(
    name = "test_base_tab",
    size = "small",
    srcs = ["test_base_tab.py"],
    deps = [
        "//nuplan/planning/metrics:metric_engine",
        "//nuplan/planning/metrics:metric_file",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/nuboard/base:base_tab",
        "//nuplan/planning/nuboard/base:data_class",
        "//nuplan/planning/nuboard/base:experiment_file_data",
        "//nuplan/planning/nuboard/base:plot_data",
        "//nuplan/planning/nuboard/utils/test:utils",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario_builder",
        "//nuplan/planning/simulation/main_callback:metric_file_callback",
        requirement("bokeh"),
    ],
)

py_test(
    name = "test_nuboard_file",
    size = "small",
    srcs = ["test_nuboard_file.py"],
    deps = [
        "//nuplan/planning/nuboard/base:data_class",
    ],
)

py_test(
    name = "test_simulation_tile",
    size = "small",
    srcs = ["test_simulation_tile.py"],
    deps = [
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/planning/nuboard/base:data_class",
        "//nuplan/planning/nuboard/base:experiment_file_data",
        "//nuplan/planning/nuboard/base:plot_data",
        "//nuplan/planning/nuboard/base:simulation_tile",
        "//nuplan/planning/nuboard/utils/test:utils",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
        requirement("bokeh"),
        requirement("hypothesis"),
    ],
)
