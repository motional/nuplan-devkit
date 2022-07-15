load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_remote_planner",
    size = "small",
    srcs = ["test_remote_planner.py"],
    deps = [
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/planner:remote_planner",
    ],
)
