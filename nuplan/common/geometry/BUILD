load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "compute",
    srcs = ["compute.py"],
    deps = [
        ":transform",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:vehicle_parameters",
    ],
)

py_library(
    name = "convert",
    srcs = ["convert.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
    ],
)

py_library(
    name = "transform",
    srcs = ["transform.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
    ],
)

py_library(
    name = "tracked_object_interpolator",
    srcs = ["tracked_object_interpolator.py"],
    deps = [
        "//nuplan/common/actor_state:agent_temporal_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:tracked_objects",
        "//nuplan/common/utils:interpolatable_state",
        "//nuplan/planning/simulation/trajectory:interpolated_trajectory",
        "//nuplan/planning/simulation/trajectory:predicted_trajectory",
    ],
)
