load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "abstract_tracker",
    srcs = ["abstract_tracker.py"],
    deps = [
        "//nuplan/common/actor_state:dynamic_car_state",
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
        "//nuplan/planning/simulation/trajectory:abstract_trajectory",
    ],
)

py_library(
    name = "lqr",
    srcs = ["lqr.py"],
    deps = [
        "//nuplan/common/actor_state:dynamic_car_state",
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/database/utils:measure",
        "//nuplan/planning/simulation/controller/tracker:abstract_tracker",
        "//nuplan/planning/simulation/controller/tracker:tracker_utils",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
        "//nuplan/planning/simulation/trajectory:abstract_trajectory",
        requirement("numpy"),
    ],
)

py_library(
    name = "tracker_utils",
    srcs = ["tracker_utils.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/geometry:compute",
        "//nuplan/planning/simulation/trajectory:abstract_trajectory",
        requirement("numpy"),
    ],
)

py_library(
    name = "ilqr_tracker",
    srcs = ["ilqr_tracker.py"],
    deps = [
        "//nuplan/common/actor_state:dynamic_car_state",
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/simulation/controller/tracker:abstract_tracker",
        "//nuplan/planning/simulation/controller/tracker/ilqr:ilqr_solver",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
        "//nuplan/planning/simulation/trajectory:abstract_trajectory",
    ],
)
