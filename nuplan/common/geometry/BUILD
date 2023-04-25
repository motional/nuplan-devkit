load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")
load("@pip_torch_deps//:requirements.bzl", requirement_torch = "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "torch_geometry",
    srcs = ["torch_geometry.py"],
    deps = [
        requirement_torch("torch"),
    ],
)

py_library(
    name = "compute",
    srcs = ["compute.py"],
    deps = [
        ":transform",
        "//nuplan/common/actor_state:oriented_box",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:vehicle_parameters",
        requirement("numpy"),
        requirement("scipy"),
        requirement("shapely"),
    ],
)

py_library(
    name = "convert",
    srcs = ["convert.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        requirement("numpy"),
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
    name = "interpolate_tracked_object",
    srcs = ["interpolate_tracked_object.py"],
    deps = [
        ":interpolate_state",
        "//nuplan/common/actor_state:agent_temporal_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:tracked_objects",
        "//nuplan/common/utils:interpolatable_state",
        "//nuplan/planning/simulation/trajectory:interpolated_trajectory",
        "//nuplan/planning/simulation/trajectory:predicted_trajectory",
    ],
)

py_library(
    name = "interpolate_state",
    srcs = ["interpolate_state.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/utils:interpolatable_state",
        "//nuplan/planning/simulation/trajectory:interpolated_trajectory",
    ],
)
