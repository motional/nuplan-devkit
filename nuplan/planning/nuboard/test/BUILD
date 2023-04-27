load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_test(
    name = "test_nuboard",
    size = "medium",
    srcs = ["test_nuboard.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/planning/nuboard",
        "//nuplan/planning/nuboard/base:data_class",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario_builder",
        requirement("hypothesis"),
    ],
)
