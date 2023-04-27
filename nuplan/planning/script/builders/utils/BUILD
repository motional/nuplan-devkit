load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")
load("@pip_torch_deps//:requirements.bzl", requirement_torch = "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "utils_checkpoint",
    srcs = ["utils_checkpoint.py"],
)

py_library(
    name = "utils_config",
    srcs = ["utils_config.py"],
    deps = [
        "//nuplan/planning/script/builders/utils:utils_type",
        "//nuplan/planning/simulation/callback:timing_callback",
        "//nuplan/planning/utils/multithreading:worker_utils",
        requirement("omegaconf"),
        requirement_torch("torch"),
        requirement_torch("tensorboard"),
        requirement_torch("pytorch-lightning"),
    ],
)

py_library(
    name = "utils_type",
    srcs = ["utils_type.py"],
    deps = [
        requirement("hydra-core"),
        requirement_torch("pytorch-lightning"),
    ],
)
