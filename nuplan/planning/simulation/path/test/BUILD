load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_path",
    size = "small",
    srcs = ["test_path.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/simulation/path:interpolated_path",
        "//nuplan/planning/simulation/path:utils",
    ],
)
