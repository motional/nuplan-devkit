load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "training",
    srcs = ["training.py"],
    deps = [
        "//nuplan/planning/scenario_builder:abstract_scenario_builder",
        "//nuplan/planning/script/builders:model_builder",
        "//nuplan/planning/script/builders:training_builder",
        "//nuplan/planning/training/callbacks:profile_callback",
        "//nuplan/planning/utils/multithreading:worker_pool",
    ],
)
