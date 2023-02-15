load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_interpolated_trajectory",
    size = "small",
    srcs = ["test_interpolated_trajectory.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/utils:helpers",
        "//nuplan/common/utils:interpolatable_state",
        "//nuplan/common/utils:split_state",
        "//nuplan/planning/simulation/trajectory:interpolated_trajectory",
    ],
)
