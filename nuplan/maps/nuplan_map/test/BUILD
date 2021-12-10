load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

filegroup(
    name = "test_json",
    srcs = glob(["json/**/*.json"]),
    visibility = ["//visibility:private"],
)

py_test(
    name = "test_nuplan_map",
    size = "large",
    srcs = ["test_nuplan_map.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/actor_state:state_representation",
        "//nuplan/database/maps_db:gpkg_mapsdb",
        "//nuplan/maps:abstract_map",
        "//nuplan/maps:abstract_map_objects",
        "//nuplan/maps:test_utils",
        "//nuplan/maps/nuplan_map",
        "//nuplan/maps/nuplan_map:map_factory",
        "//nuplan/utils/testing:nuplan_test",
    ],
)

py_test(
    name = "test_map_objects",
    size = "large",
    srcs = ["test_map_objects.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/actor_state:state_representation",
        "//nuplan/database/maps_db:gpkg_mapsdb",
        "//nuplan/maps:abstract_map",
        "//nuplan/maps:abstract_map_objects",
        "//nuplan/maps:test_utils",
        "//nuplan/maps/nuplan_map",
        "//nuplan/maps/nuplan_map:crosswalk",
        "//nuplan/maps/nuplan_map:intersection",
        "//nuplan/maps/nuplan_map:map_factory",
        "//nuplan/maps/nuplan_map:stop_line",
        "//nuplan/utils/testing:nuplan_test",
    ],
)
