load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "color",
    srcs = ["color.py"],
    deps = [
        requirement("numpy"),
    ],
)
