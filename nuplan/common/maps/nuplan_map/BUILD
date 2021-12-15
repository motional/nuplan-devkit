load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "baseline_path",
    srcs = ["baseline_path.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map_objects",
    ],
)

py_library(
    name = "map_objects",
    srcs = [
        "lane.py",
        "lane_connector.py",
    ],
    deps = [
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/common/maps/nuplan_map:baseline_path",
        "//nuplan/common/maps/nuplan_map:utils",
    ],
)

py_library(
    name = "nuplan_map",
    srcs = ["nuplan_map.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/common/maps/nuplan_map:crosswalk",
        "//nuplan/common/maps/nuplan_map:intersection",
        "//nuplan/common/maps/nuplan_map:map_objects",
        "//nuplan/common/maps/nuplan_map:stop_line",
        "//nuplan/common/maps/nuplan_map:utils",
        "//nuplan/database/maps_db:imapsdb",
        "//nuplan/database/maps_db:layer",
        "//nuplan/database/maps_db:utils",
        "//nuplan/database/nuplan_db:utils",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/database/maps_db:layer",
    ],
)

py_library(
    name = "map_factory",
    srcs = ["map_factory.py"],
    deps = [
        "//nuplan/common/maps:abstract_map_factory",
        "//nuplan/common/maps/nuplan_map",
        "//nuplan/database/maps_db:imapsdb",
    ],
)

py_library(
    name = "stop_line",
    srcs = ["stop_line.py"],
    deps = [
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/common/maps/nuplan_map:utils",
    ],
)

py_library(
    name = "crosswalk",
    srcs = ["crosswalk.py"],
    deps = [
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/common/maps/nuplan_map:utils",
    ],
)

py_library(
    name = "intersection",
    srcs = ["intersection.py"],
    deps = [
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/common/maps/nuplan_map:utils",
    ],
)
