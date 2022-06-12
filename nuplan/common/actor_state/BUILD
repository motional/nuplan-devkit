load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "state_representation",
    srcs = ["state_representation.py"],
)

py_library(
    name = "transform_state",
    srcs = ["transform_state.py"],
    deps = [
        ":state_representation",
        ":vehicle_parameters",
        "//nuplan/common/geometry:transform",
    ],
)

py_library(
    name = "vehicle_parameters",
    srcs = ["vehicle_parameters.py"],
)

py_library(
    name = "oriented_box",
    srcs = ["oriented_box.py"],
    deps = [
        ":state_representation",
        "//nuplan/common/geometry:transform",
    ],
)

py_library(
    name = "car_footprint",
    srcs = ["car_footprint.py"],
    deps = [
        ":oriented_box",
        ":state_representation",
        ":vehicle_parameters",
        "//nuplan/common/geometry:transform",
    ],
)

py_library(
    name = "ego_state",
    srcs = ["ego_state.py"],
    deps = [
        ":agent",
        ":car_footprint",
        ":dynamic_car_state",
        ":state_representation",
        ":tracked_objects_types",
        "//nuplan/common/utils:interpolatable_state",
        "//nuplan/common/utils:split_state",
    ],
)

py_library(
    name = "scene_object",
    srcs = ["scene_object.py"],
    deps = [
        ":oriented_box",
        ":state_representation",
        ":tracked_objects_types",
    ],
)

py_library(
    name = "static_object",
    srcs = ["static_object.py"],
    deps = [
        ":oriented_box",
        ":state_representation",
    ],
)

py_library(
    name = "agent",
    srcs = ["agent.py"],
    deps = [
        ":oriented_box",
        ":scene_object",
        ":state_representation",
        ":tracked_objects_types",
        "//nuplan/planning/simulation/trajectory:predicted_trajectory",
    ],
)

py_library(
    name = "tracked_objects_types",
    srcs = ["tracked_objects_types.py"],
    deps = [],
)

py_library(
    name = "tracked_objects",
    srcs = ["tracked_objects.py"],
    deps = [
        ":agent",
        ":oriented_box",
        ":scene_object",
        ":static_object",
        ":tracked_objects_types",
    ],
)

py_library(
    name = "dynamic_car_state",
    srcs = ["dynamic_car_state.py"],
    deps = [
        ":state_representation",
    ],
)

py_library(
    name = "waypoint",
    srcs = ["waypoint.py"],
    deps = [
        "state_representation",
        ":oriented_box",
        "//nuplan/common/utils:interpolatable_state",
        "//nuplan/common/utils:split_state",
    ],
)
