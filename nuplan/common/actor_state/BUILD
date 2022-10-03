load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "state_representation",
    srcs = ["state_representation.py"],
    deps = [
        requirement("numpy"),
    ],
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
        requirement("shapely"),
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
        ":agent_state",
        ":car_footprint",
        ":dynamic_car_state",
        ":scene_object",
        ":state_representation",
        ":waypoint",
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
    name = "agent_state",
    srcs = ["agent_state.py"],
    deps = [
        "//nuplan/common/actor_state:oriented_box",
        "//nuplan/common/actor_state:scene_object",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:tracked_objects_types",
    ],
)

py_library(
    name = "agent",
    srcs = ["agent.py"],
    deps = [
        "//nuplan/common/actor_state:agent_state",
        "//nuplan/common/actor_state:agent_temporal_state",
        "//nuplan/common/actor_state:oriented_box",
        "//nuplan/common/actor_state:scene_object",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:tracked_objects_types",
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
        ":agent_temporal_state",
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

py_library(
    name = "agent_temporal_state",
    srcs = ["agent_temporal_state.py"],
    deps = [
        "//nuplan/common/actor_state:oriented_box",
        "//nuplan/common/actor_state:scene_object",
        "//nuplan/common/actor_state:tracked_objects_types",
        "//nuplan/planning/simulation/trajectory:predicted_trajectory",
    ],
)

py_library(
    name = "ego_temporal_state",
    srcs = ["ego_temporal_state.py"],
    deps = [
        "//nuplan/common/actor_state:agent_temporal_state",
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:tracked_objects_types",
        "//nuplan/planning/simulation/trajectory:predicted_trajectory",
    ],
)
