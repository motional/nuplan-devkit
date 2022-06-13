load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_kinematic_motion_model",
    size = "small",
    srcs = ["test_kinematic_motion_model.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/common/actor_state/test:test_utils",
        "//nuplan/planning/simulation/controller/motion_model:kinematic_bicycle",
    ],
)
