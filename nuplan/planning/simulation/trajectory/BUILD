load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "interpolated",
    srcs = ["interpolated.py"],
    deps = [
        "//nuplan/actor_state:ego_state",
        "//nuplan/actor_state:state_representation",
        "//nuplan/planning/simulation/trajectory",
    ],
)

py_library(
    name = "trajectory",
    srcs = ["trajectory.py"],
    deps = [
        "//nuplan/actor_state:agent",
        "//nuplan/actor_state:ego_state",
        "//nuplan/actor_state:state_representation",
    ],
)

py_library(
    name = "trajectory_sampling",
    srcs = ["trajectory_sampling.py"],
)
