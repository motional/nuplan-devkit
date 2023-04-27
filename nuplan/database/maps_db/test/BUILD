load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

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
        requirement("numpy"),
    ],
)

py_test(
    name = "test_map_api",
    size = "large",
    srcs = ["test_map_api.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/database/maps_db:gpkg_mapsdb",
        "//nuplan/database/maps_db:map_api",
        "//nuplan/database/maps_db:map_explorer",
        "//nuplan/database/tests:test_utils_nuplan_db",
    ],
)

py_test(
    name = "test_map_explorer",
    size = "small",
    srcs = ["test_map_explorer.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/database/maps_db:gpkg_mapsdb",
        "//nuplan/database/maps_db:map_api",
        "//nuplan/database/maps_db:map_explorer",
        "//nuplan/database/tests:test_utils_nuplan_db",
    ],
)
