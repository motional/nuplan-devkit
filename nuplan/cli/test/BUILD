load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_nuplan_cli",
    size = "small",
    srcs = ["test_nuplan_cli.py"],
    deps = [
        "//nuplan/cli:nuplan_cli",
        "//nuplan/common/actor_state:oriented_box",
        "//nuplan/common/actor_state:state_representation",
    ],
)
