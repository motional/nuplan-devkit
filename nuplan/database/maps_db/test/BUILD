load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_test(
    name = "test_layer",
    size = "small",
    srcs = ["test_layer.py"],
    deps = [
        "//nuplan/database/maps_db:layer",
        "//nuplan/database/maps_db:metadata",
        "//nuplan/database/maps_db:utils",
    ],
)

py_test(
    name = "test_map_api",
    size = "large",
    srcs = ["test_map_api.py"],
    deps = [
        "//nuplan/database/maps_db:gpkg_mapsdb",
        "//nuplan/database/maps_db:map_api",
        "//nuplan/database/maps_db:map_explorer",
    ],
)

py_test(
    name = "test_map_explorer",
    size = "small",
    srcs = ["test_map_explorer.py"],
    deps = [
        "//nuplan/database/maps_db:gpkg_mapsdb",
        "//nuplan/database/maps_db:map_api",
        "//nuplan/database/maps_db:map_explorer",
    ],
)
