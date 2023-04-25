load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "iterable_lidar_box",
    srcs = ["iterable_lidar_box.py"],
)

py_library(
    name = "geometry",
    srcs = ["geometry.py"],
    deps = [
        requirement("scipy"),
    ],
)

py_library(
    name = "measure",
    srcs = ["measure.py"],
    deps = [
        "//nuplan/database/utils:geometry",
        "//nuplan/database/utils/boxes:box3d",
        requirement("scipy"),
        requirement("shapely"),
    ],
)

py_library(
    name = "plot",
    srcs = ["plot.py"],
    deps = [
        requirement("matplotlib"),
    ],
)

py_library(
    name = "image",
    srcs = ["image.py"],
    deps = [
        requirement("opencv-python"),
        requirement("numpy"),
        requirement("Pillow"),
    ],
)
