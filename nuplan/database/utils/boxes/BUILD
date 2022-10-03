load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "box",
    srcs = ["box.py"],
)

py_library(
    name = "box3d",
    srcs = ["box3d.py"],
    deps = [
        "//nuplan/database/utils:geometry",
        "//nuplan/database/utils:plot",
        "//nuplan/database/utils/boxes:box",
        "//nuplan/database/utils/label",
        requirement("opencv-python"),
        requirement("matplotlib"),
        requirement("pyquaternion"),
    ],
)
