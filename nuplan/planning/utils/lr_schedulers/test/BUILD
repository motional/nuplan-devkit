load("@rules_python//python:defs.bzl", "py_test")
load("@pip_torch_deps//:requirements.bzl", requirement_torch = "requirement")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_sequential_lr_scheduler",
    size = "small",
    srcs = ["test_sequential_lr_scheduler.py"],
    deps = [
        "//nuplan/planning/utils/lr_schedulers:sequential_lr_scheduler",
        requirement_torch("torch"),
    ],
)
