load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "abstract_map",
    srcs = ["abstract_map.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/common/maps:maps_datatypes",
    ],
)

py_library(
    name = "maps_datatypes",
    srcs = ["maps_datatypes.py"],
    deps = [
        requirement("geopandas"),
    ],
)

py_library(
    name = "abstract_map_factory",
    srcs = ["abstract_map_factory.py"],
    deps = ["//nuplan/common/maps:abstract_map"],
)

py_library(
    name = "abstract_map_objects",
    srcs = ["abstract_map_objects.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:maps_datatypes",
    ],
)

py_library(
    name = "map_manager",
    srcs = ["map_manager.py"],
    deps = [
        ":abstract_map",
        ":abstract_map_factory",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/common/maps:maps_datatypes",
    ],
)
