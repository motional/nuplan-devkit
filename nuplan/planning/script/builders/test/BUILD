load("@rules_python//python:defs.bzl", "py_test")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_log",
    size = "medium",
    srcs = ["test_log.py"],
    deps = [
        "//nuplan/planning/script/builders:logging_builder",
    ],
)

py_test(
    name = "test_scenario_builder",
    size = "medium",
    srcs = ["test_scenario_builder.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/planning/script/builders:scenario_builder",
        "//nuplan/planning/training/experiments:cache_metadata_entry",
        "//nuplan/planning/training/modeling:torch_module_wrapper",
        "//nuplan/planning/training/modeling:types",
        "//nuplan/planning/training/preprocessing/feature_builders:abstract_feature_builder",
        "//nuplan/planning/utils/multithreading:worker_pool",
        requirement("mock"),
    ],
)

py_test(
    name = "test_lr_scheduler_builder",
    size = "small",
    srcs = ["test_lr_scheduler_builder.py"],
    deps = [
        "//nuplan/planning/script/builders:lr_scheduler_builder",
    ],
)

py_test(
    name = "test_simulation_callback_builder",
    size = "small",
    srcs = ["test_simulation_callback_builder.py"],
    tags = [],
    deps = [
        "//nuplan/planning/script/builders:simulation_callback_builder",
        "//nuplan/planning/simulation/callback:metric_callback",
        "//nuplan/planning/simulation/callback:multi_callback",
        "//nuplan/planning/simulation/callback:serialization_callback",
        "//nuplan/planning/simulation/callback:timing_callback",
        "//nuplan/planning/simulation/callback:visualization_callback",
        "//nuplan/planning/utils/multithreading:worker_parallel",
        "//nuplan/planning/utils/multithreading:worker_sequential",
        requirement("hypothesis"),
        requirement("omegaconf"),
    ],
)
