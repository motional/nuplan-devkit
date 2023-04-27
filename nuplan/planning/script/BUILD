load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")
load("@pip_torch_deps//:requirements.bzl", requirement_torch = "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

filegroup(
    name = "yaml",
    srcs = glob(["**/*.yaml"]),
)

filegroup(
    name = "init",
    srcs = glob(["**/__init__.py"]),
)

py_binary(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//nuplan/common/utils:file_backed_barrier",
        "//nuplan/common/utils:io_utils",
        "//nuplan/planning/script/builders:folder_builder",
        "//nuplan/planning/script/builders:logging_builder",
        "//nuplan/planning/script/builders:main_callback_builder",
        "//nuplan/planning/script/builders:worker_pool_builder",
        "//nuplan/planning/script/builders/utils:utils_config",
        "//nuplan/planning/simulation/main_callback:multi_main_callback",
        "//nuplan/planning/simulation/runner:abstract_runner",
        "//nuplan/planning/simulation/runner:executor",
        "//nuplan/planning/simulation/runner:runner_report",
        "//nuplan/planning/training/callbacks:profile_callback",
        "//nuplan/planning/utils/multithreading:worker_pool",
        requirement("s3fs"),
    ],
)

py_binary(
    name = "run_simulation",
    srcs = ["run_simulation.py"],
    data = [
        ":init",
        ":yaml",
    ],
    deps = [
        "//nuplan/common/utils:s3_utils",
        "//nuplan/planning/script:utils",
        "//nuplan/planning/script/builders:simulation_builder",
        "//nuplan/planning/script/builders:simulation_callback_builder",
        "//nuplan/planning/simulation/planner:abstract_planner",
        requirement_torch("pytorch-lightning"),
    ],
)

py_binary(
    name = "run_metric_aggregator",
    srcs = ["run_metric_aggregator.py"],
    data = [
        ":init",
        ":yaml",
    ],
    deps = [
        ":run_simulation",
        "//nuplan/planning/script:utils",
        "//nuplan/planning/script/builders:metric_aggregator_builder",
        "//nuplan/planning/simulation/main_callback:metric_aggregator_callback",
        "//nuplan/planning/simulation/main_callback:metric_summary_callback",
    ],
)

py_binary(
    name = "run_metric",
    srcs = ["run_metric.py"],
    data = [
        ":init",
        ":yaml",
    ],
    deps = [
        "//nuplan/planning/script:run_simulation",
        "//nuplan/planning/script:utils",
        "//nuplan/planning/script/builders:metric_runner_builder",
        "//nuplan/planning/script/builders:simulation_log_builder",
    ],
)

py_binary(
    name = "run_training",
    srcs = ["run_training.py"],
    data = [
        ":init",
        ":yaml",
        requirement_torch("pytorch-lightning"),
    ],
    deps = [
        ":profiler_context_manager",
        "//nuplan/common/utils:s3_utils",
        "//nuplan/planning/script:utils",
        "//nuplan/planning/script/builders:folder_builder",
        "//nuplan/planning/script/builders:logging_builder",
        "//nuplan/planning/script/builders:scenario_building_builder",
        "//nuplan/planning/script/builders:worker_pool_builder",
        "//nuplan/planning/script/builders/utils:utils_config",
        "//nuplan/planning/training/experiments:caching",
        "//nuplan/planning/training/experiments:training",
    ],
)

py_binary(
    name = "run_nuboard",
    srcs = ["run_nuboard.py"],
    data = [
        ":init",
        ":yaml",
    ],
    deps = [
        "//nuplan/planning/nuboard",
        "//nuplan/planning/script:utils",
        "//nuplan/planning/script/builders:scenario_building_builder",
        "//nuplan/planning/script/builders/utils:utils_config",
        requirement("nest_asyncio"),
    ],
)

py_binary(
    name = "run_result_processor_aws",
    srcs = ["run_result_processor_aws.py"],
    data = [
        ":init",
        ":yaml",
    ],
    deps = [
        ":run_metric_aggregator",
        ":run_simulation",
        "//nuplan/planning/script:utils",
        "//nuplan/planning/simulation/main_callback:publisher_callback",
        "//nuplan/submission/evalai:leaderboard_writer",
        "//nuplan/submission/utils",
        "//nuplan/submission/utils:aws_utils",
    ],
)

py_binary(
    name = "run_submission_planner",
    srcs = ["run_submission_planner.py"],
    deps = [
        "//nuplan/planning/script:utils",
        "//nuplan/submission:submission_planner",
    ],
)

py_binary(
    name = "profiler_context_manager",
    srcs = ["profiler_context_manager.py"],
    deps = [
        "//nuplan/planning/training/callbacks:profile_callback",
    ],
)
