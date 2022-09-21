load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

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
        requirement("torch"),
        requirement("tensorboard"),
        requirement("pytorch-lightning"),
    ],
)

py_library(
    name = "utils_type",
    srcs = ["utils_type.py"],
    deps = [
        requirement("hydra-core"),
        requirement("pytorch-lightning"),
    ],
)
