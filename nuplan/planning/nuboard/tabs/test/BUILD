load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "skeleton_test_tab",
    srcs = ["skeleton_test_tab.py"],
    deps = [
        "//nuplan/planning/metrics:metric_engine",
        "//nuplan/planning/metrics:metric_file",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/nuboard/base:data_class",
        "//nuplan/planning/nuboard/base:experiment_file_data",
        "//nuplan/planning/nuboard/utils/test:utils",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario_builder",
        "//nuplan/planning/simulation:simulation_log",
        "//nuplan/planning/simulation/main_callback:metric_file_callback",
    ],
)

py_test(
    name = "test_configuration_tab",
    size = "medium",
    srcs = ["test_configuration_tab.py"],
    deps = [
        "//nuplan/planning/nuboard/base:data_class",
        "//nuplan/planning/nuboard/base:experiment_file_data",
        "//nuplan/planning/nuboard/tabs:configuration_tab",
        "//nuplan/planning/nuboard/tabs:histogram_tab",
        requirement("bokeh"),
    ],
)

py_test(
    name = "test_histogram_tab",
    size = "medium",
    srcs = ["test_histogram_tab.py"],
    deps = [
        ":skeleton_test_tab",
        "//nuplan/planning/nuboard/base:experiment_file_data",
        "//nuplan/planning/nuboard/tabs:histogram_tab",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario_builder",
        "//nuplan/planning/simulation:simulation_log",
        requirement("bokeh"),
    ],
)

py_test(
    name = "test_overview_tab",
    size = "medium",
    srcs = ["test_overview_tab.py"],
    deps = [
        ":skeleton_test_tab",
        "//nuplan/planning/nuboard/base:experiment_file_data",
        "//nuplan/planning/nuboard/tabs:overview_tab",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario_builder",
        "//nuplan/planning/simulation:simulation_log",
        requirement("bokeh"),
    ],
)

py_test(
    name = "test_scenario_tab",
    size = "medium",
    srcs = ["test_scenario_tab.py"],
    tags = ["integration"],
    deps = [
        ":skeleton_test_tab",
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/planning/nuboard/base:experiment_file_data",
        "//nuplan/planning/nuboard/tabs:scenario_tab",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario_builder",
        requirement("bokeh"),
    ],
)

py_test(
    name = "test_cloud_tab",
    size = "medium",
    srcs = ["test_cloud_tab.py"],
    deps = [
        "//nuplan/planning/nuboard/base:data_class",
        "//nuplan/planning/nuboard/base:experiment_file_data",
        "//nuplan/planning/nuboard/tabs:cloud_tab",
        "//nuplan/planning/nuboard/tabs:configuration_tab",
        "//nuplan/planning/nuboard/tabs:histogram_tab",
        requirement("bokeh"),
    ],
)
