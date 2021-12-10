load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "raster_visualization",
    srcs = ["raster_visualization.py"],
    deps = [
        "//nuplan/planning/training/preprocessing/features:raster",
        "//nuplan/planning/training/preprocessing/features:trajectory",
    ],
)

py_library(
    name = "visualization_callbacks",
    srcs = ["visualization_callbacks.py"],
    deps = [
        "//nuplan/planning/training/modeling:types",
        "//nuplan/planning/training/preprocessing:feature_collate",
        "//nuplan/planning/training/preprocessing/features:raster",
        "//nuplan/planning/training/preprocessing/features:trajectory",
        "//nuplan/planning/training/visualization:raster_visualization",
    ],
)
