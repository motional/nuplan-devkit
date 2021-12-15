load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "interpolated_path",
    srcs = ["interpolated_path.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/simulation/path",
    ],
)

py_library(
    name = "path",
    srcs = ["path.py"],
    deps = ["//nuplan/common/actor_state:state_representation"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/simulation/path",
    ],
)
