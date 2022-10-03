load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "abstract_submission_validator",
    srcs = ["abstract_submission_validator.py"],
)

py_library(
    name = "base_submission_validator",
    srcs = ["base_submission_validator.py"],
    deps = [
        ":abstract_submission_validator",
    ],
)

py_library(
    name = "image_exists_validator",
    srcs = ["image_exists_validator.py"],
    deps = [
        ":base_submission_validator",
        requirement("docker"),
    ],
)

py_library(
    name = "image_is_runnable_validator",
    srcs = ["image_is_runnable_validator.py"],
    deps = [
        ":base_submission_validator",
        "//nuplan/submission:submission_container",
        "//nuplan/submission/utils",
        requirement("docker"),
    ],
)

py_library(
    name = "submission_computes_trajectory_validator",
    srcs = ["submission_computes_trajectory_validator.py"],
    deps = [
        ":base_submission_validator",
        "//nuplan/planning/scenario_builder/nuplan_db/test:nuplan_scenario_test_utils",
        "//nuplan/planning/simulation/history:simulation_history_buffer",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/planner:remote_planner",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
        "//nuplan/submission:submission_container_factory",
        "//nuplan/submission:submission_container_manager",
        "//nuplan/submission/utils",
    ],
)
