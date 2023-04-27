load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "metadata",
    srcs = ["metadata.py"],
)

py_library(
    name = "layer",
    srcs = ["layer.py"],
    deps = [
        "//nuplan/database/maps_db:metadata",
    ],
)

py_library(
    name = "imapsdb",
    srcs = ["imapsdb.py"],
    deps = [
        "//nuplan/database/maps_db:layer",
    ],
)

py_library(
    name = "layer_dataset_ops",
    srcs = ["layer_dataset_ops.py"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//nuplan/database/maps_db:metadata",
        requirement("opencv-python"),
        requirement("geopandas"),
    ],
)

py_library(
    name = "gpkg_mapsdb",
    srcs = ["gpkg_mapsdb.py"],
    deps = [
        "//nuplan/common/utils:s3_utils",
        "//nuplan/database/common/blob_store:creator",
        "//nuplan/database/maps_db:imapsdb",
        "//nuplan/database/maps_db:layer",
        "//nuplan/database/maps_db:layer_dataset_ops",
        "//nuplan/database/maps_db:metadata",
        requirement("rasterio"),
        requirement("fiona"),
        requirement("pyogrio"),
        requirement("geopandas"),
    ],
)

py_library(
    name = "map_api",
    srcs = ["map_api.py"],
    deps = [
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/common/maps/nuplan_map",
        "//nuplan/database/maps_db:gpkg_mapsdb",
        "//nuplan/database/maps_db:layer_dataset_ops",
    ],
)

py_library(
    name = "map_explorer",
    srcs = ["map_explorer.py"],
    deps = [
        "//nuplan/database/maps_db:map_api",
    ],
)
