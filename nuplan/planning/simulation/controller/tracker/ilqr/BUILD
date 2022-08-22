load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "ilqr_solver",
    srcs = ["ilqr_solver.py"],
    deps = [
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/common/geometry:compute",
        "//nuplan/planning/simulation/controller/tracker:tracker_utils",
    ],
)
