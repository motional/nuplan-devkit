load("@rules_python//python:defs.bzl", "py_test")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_ego_trajectory_target_builder",
    size = "medium",
    srcs = ["test_ego_trajectory_target_builder.py"],
    tags = ["manual"],
    deps = [
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
        "//nuplan/planning/simulation/trajectory:trajectory_sampling",
        "//nuplan/planning/training/preprocessing/target_builders:ego_trajectory_target_builder",
        requirement("hypothesis"),
        requirement("numpy"),
    ],
)
