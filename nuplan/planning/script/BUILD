load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

filegroup(
    name = "yaml",
    srcs = glob(["**/*.yaml"]),
)

py_binary(
    name = "default_path",
    srcs = ["default_path.py"],
    deps = [],
)

py_binary(
    name = "run_simulation",
    srcs = ["run_simulation.py"],
    data = [
        "//nuplan/planning/script:yaml",
    ],
    deps = [
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/planning/nuboard",
        "//nuplan/planning/script:default_path",
        "//nuplan/planning/script/builders:folder_builder",
        "//nuplan/planning/script/builders:logging_builder",
        "//nuplan/planning/script/builders:metric_builder",
        "//nuplan/planning/script/builders:scenario_building_builder",
        "//nuplan/planning/script/builders:simulation_builder",
        "//nuplan/planning/script/builders:simulation_callback_builder",
        "//nuplan/planning/script/builders:worker_pool_builder",
        "//nuplan/planning/script/builders/utils:utils_config",
        "//nuplan/planning/simulation:runner",
    ],
)

py_binary(
    name = "run_training",
    srcs = ["run_training.py"],
    data = [
        "//nuplan/planning/script:yaml",
    ],
    deps = [
        "//nuplan/planning/script:default_path",
        "//nuplan/planning/script/builders:folder_builder",
        "//nuplan/planning/script/builders:logging_builder",
        "//nuplan/planning/script/builders:scenario_building_builder",
        "//nuplan/planning/script/builders:worker_pool_builder",
        "//nuplan/planning/script/builders/utils:utils_config",
        "//nuplan/planning/training/experiments:training",
    ],
)

py_binary(
    name = "run_nuboard",
    srcs = ["run_nuboard.py"],
    data = [
        "//nuplan/planning/script:yaml",
    ],
    deps = [
        "//nuplan/planning/nuboard",
        "//nuplan/planning/script:default_path",
        "//nuplan/planning/script/builders:metric_builder",
        "//nuplan/planning/script/builders:scenario_building_builder",
        "//nuplan/planning/script/builders/utils:utils_config",
    ],
)
