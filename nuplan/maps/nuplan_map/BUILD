load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "baseline_path",
    srcs = ["baseline_path.py"],
    deps = [
        "//nuplan/actor_state:state_representation",
        "//nuplan/maps:abstract_map_objects",
    ],
)

py_library(
    name = "map_objects",
    srcs = [
        "lane.py",
        "lane_connector.py",
    ],
    deps = [
        "//nuplan/maps:abstract_map_objects",
        "//nuplan/maps:maps_datatypes",
        "//nuplan/maps/nuplan_map:baseline_path",
        "//nuplan/maps/nuplan_map:utils",
    ],
)

py_library(
    name = "nuplan_map",
    srcs = ["nuplan_map.py"],
    deps = [
        "//nuplan/actor_state:state_representation",
        "//nuplan/database/maps_db:imapsdb",
        "//nuplan/database/maps_db:layer",
        "//nuplan/database/maps_db:utils",
        "//nuplan/database/nuplan_db:utils",
        "//nuplan/maps:abstract_map",
        "//nuplan/maps:maps_datatypes",
        "//nuplan/maps/nuplan_map:crosswalk",
        "//nuplan/maps/nuplan_map:intersection",
        "//nuplan/maps/nuplan_map:map_objects",
        "//nuplan/maps/nuplan_map:stop_line",
        "//nuplan/maps/nuplan_map:utils",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//nuplan/actor_state:state_representation",
        "//nuplan/database/maps_db:layer",
        "//nuplan/maps:maps_datatypes",
    ],
)

py_library(
    name = "map_factory",
    srcs = ["map_factory.py"],
    deps = [
        "//nuplan/database/maps_db:imapsdb",
        "//nuplan/maps:abstract_map_factory",
        "//nuplan/maps/nuplan_map",
    ],
)

py_library(
    name = "stop_line",
    srcs = ["stop_line.py"],
    deps = [
        "//nuplan/maps:abstract_map_objects",
        "//nuplan/maps:maps_datatypes",
        "//nuplan/maps/nuplan_map:utils",
    ],
)

py_library(
    name = "crosswalk",
    srcs = ["crosswalk.py"],
    deps = [
        "//nuplan/maps:abstract_map_objects",
        "//nuplan/maps:maps_datatypes",
        "//nuplan/maps/nuplan_map:utils",
    ],
)

py_library(
    name = "intersection",
    srcs = ["intersection.py"],
    deps = [
        "//nuplan/maps:abstract_map_objects",
        "//nuplan/maps:maps_datatypes",
        "//nuplan/maps/nuplan_map:utils",
    ],
)
