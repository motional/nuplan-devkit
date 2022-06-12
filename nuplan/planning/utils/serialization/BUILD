load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "from_scene",
    srcs = ["from_scene.py"],
    deps = [
        ":scene_simple_trajectory",
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/common/geometry:transform",
        "//nuplan/database/utils/boxes:box3d",
    ],
)

py_library(
    name = "scene_simple_trajectory",
    srcs = ["scene_simple_trajectory.py"],
    deps = [
        "@nuplan_devkit//nuplan/common/actor_state:oriented_box",
        "@nuplan_devkit//nuplan/common/actor_state:state_representation",
        "@nuplan_devkit//nuplan/common/actor_state:waypoint",
        "@nuplan_devkit//nuplan/planning/simulation/trajectory:abstract_trajectory",
    ],
)

py_library(
    name = "to_scene",
    srcs = ["to_scene.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:tracked_objects",
        "//nuplan/common/actor_state:tracked_objects_types",
        "//nuplan/planning/simulation/trajectory:abstract_trajectory",
    ],
)
