load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "evalai_interface",
    srcs = ["evalai_interface.py"],
    deps = [
        requirement("requests"),
    ],
)

py_library(
    name = "leaderboard_writer",
    srcs = ["leaderboard_writer.py"],
    deps = [
        ":evalai_interface",
    ],
)
