load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "abstract_motion_model",
    srcs = ["abstract_motion_model.py"],
    deps = [
        "//nuplan/common/actor_state:dynamic_car_state",
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
    ],
)

py_library(
    name = "kinematic_bicycle",
    srcs = ["kinematic_bicycle.py"],
    deps = [
        "//nuplan/common/actor_state:dynamic_car_state",
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/common/geometry:compute",
        "//nuplan/planning/simulation/controller:utils",
        "//nuplan/planning/simulation/controller/motion_model:abstract_motion_model",
    ],
)
