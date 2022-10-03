load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "interpolated_trajectory",
    srcs = ["interpolated_trajectory.py"],
    deps = [
        ":abstract_trajectory",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/geometry:compute",
        "//nuplan/common/utils:interpolatable_state",
        requirement("scipy"),
    ],
)

py_library(
    name = "abstract_trajectory",
    srcs = ["abstract_trajectory.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
    ],
)

py_library(
    name = "trajectory_sampling",
    srcs = ["trajectory_sampling.py"],
)

py_library(
    name = "predicted_trajectory",
    srcs = ["predicted_trajectory.py"],
    deps = [
        ":abstract_trajectory",
        ":interpolated_trajectory",
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:waypoint",
    ],
)
