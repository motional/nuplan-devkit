load("@rules_python//python:defs.bzl", "py_test")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_utils_checkpoint",
    size = "small",
    srcs = ["test_utils_checkpoint.py"],
    deps = [
        "//nuplan/planning/script/builders/utils:utils_checkpoint",
    ],
)

py_test(
    name = "test_utils_config",
    size = "small",
    srcs = ["test_utils_config.py"],
    tags = [],
    deps = [
        "//nuplan/common/utils/test_utils:patch",
        "//nuplan/planning/script/builders/utils:utils_config",
        "//nuplan/planning/simulation/callback:metric_callback",
        "//nuplan/planning/simulation/callback:simulation_log_callback",
        "//nuplan/planning/simulation/callback:timing_callback",
        requirement("hypothesis"),
        requirement("omegaconf"),
    ],
)

py_test(
    name = "test_utils_config_distributed_training",
    size = "small",
    srcs = ["test_utils_config_distributed_training.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/planning/script/builders/utils:utils_config",
        requirement("omegaconf"),
    ],
)

py_test(
    name = "test_utils_type",
    size = "small",
    srcs = ["test_utils_type.py"],
    deps = [
        "//nuplan/planning/script/builders/utils:utils_type",
        requirement("hydra-core"),
    ],
)
