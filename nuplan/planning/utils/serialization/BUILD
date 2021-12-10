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
        "//nuplan/actor_state:ego_state",
        "//nuplan/actor_state:state_representation",
        "//nuplan/actor_state:transform_state",
        "//nuplan/actor_state:vehicle_parameters",
        "//nuplan/database/utils/boxes:box3d",
    ],
)

py_library(
    name = "to_scene",
    srcs = ["to_scene.py"],
    deps = [
        "//nuplan/actor_state:agent",
        "//nuplan/actor_state:ego_state",
        "//nuplan/actor_state:state_representation",
        "//nuplan/actor_state:transform_state",
        "//nuplan/actor_state:vehicle_parameters",
        "//nuplan/database/utils/boxes:box3d",
        "//nuplan/database/utils/label:utils",
        "//nuplan/planning/simulation/trajectory",
    ],
)
