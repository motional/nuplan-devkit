load("@rules_python//python:defs.bzl", "py_test")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_evalai_interface",
    size = "small",
    srcs = ["test_evalai_interface.py"],
    deps = [
        "//nuplan/submission/evalai:evalai_interface",
    ],
)

py_test(
    name = "test_leaderboard_writer",
    size = "small",
    srcs = ["test_leaderboard_writer.py"],
    deps = [
        "//nuplan/submission/evalai:leaderboard_writer",
        requirement("hydra-core"),
        requirement("pandas"),
    ],
)
