load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "evalai_interface",
    srcs = ["evalai_interface.py"],
)

py_library(
    name = "leaderboard_writer",
    srcs = ["leaderboard_writer.py"],
    deps = [
        ":evalai_interface",
    ],
)
