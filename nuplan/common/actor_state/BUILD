load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
)

py_library(
    name = "state_representation",
    srcs = ["state_representation.py"],
)

py_library(
    name = "transform_state",
    srcs = ["transform_state.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/database/utils/boxes:box3d",
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
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:transform_state",
        "//nuplan/common/actor_state:utils",
    ],
)

py_library(
    name = "car_footprint",
    srcs = ["car_footprint.py"],
    deps = [
        "//nuplan/common/actor_state:oriented_box",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:transform_state",
        "//nuplan/common/actor_state:vehicle_parameters",
    ],
)

py_library(
    name = "ego_state",
    srcs = ["ego_state.py"],
    deps = [
        "//nuplan/common/actor_state:agent",
        "//nuplan/common/actor_state:car_footprint",
        "//nuplan/common/actor_state:state_representation",
    ],
)

py_library(
    name = "agent",
    srcs = ["agent.py"],
    deps = [
        "//nuplan/common/actor_state:oriented_box",
        "//nuplan/common/actor_state:state_representation",
    ],
)

py_library(
    name = "tracked_objects",
    srcs = ["tracked_objects.py"],
    deps = [
        "//nuplan/common/actor_state:agent",
        "//nuplan/common/actor_state:utils",
    ],
)
