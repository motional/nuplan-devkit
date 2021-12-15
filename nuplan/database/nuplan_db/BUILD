load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "frame",
    srcs = ["frame.py"],
)

py_binary(
    name = "make_readme",
    srcs = ["make_readme.py"],
    deps = [
        "//nuplan/database/common:templates",
        "//nuplan/database/common:utils",
    ],
)

py_library(
    name = "models",
    srcs = ["models.py"],
    deps = [
        "//nuplan/common/actor_state:agent",
        "//nuplan/common/actor_state:oriented_box",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/database/common:data_types",
        "//nuplan/database/common:db",
        "//nuplan/database/common:sql_types",
        "//nuplan/database/common:utils",
        "//nuplan/database/maps_db:layer",
        "//nuplan/database/maps_db:utils",
        "//nuplan/database/nuplan_db:frame",
        "//nuplan/database/nuplan_db:utils",
        "//nuplan/database/utils:geometry",
        "//nuplan/database/utils/boxes:box3d",
        "//nuplan/database/utils/label:utils",
        "//nuplan/database/utils/pointclouds:lidar",
    ],
)

py_library(
    name = "nuplandb",
    srcs = ["nuplandb.py"],
    deps = [
        "//nuplan/database/common:db",
        "//nuplan/database/common/blob_store",
        "//nuplan/database/common/blob_store:creator",
        "//nuplan/database/maps_db:gpkg_mapsdb",
        "//nuplan/database/nuplan_db:models",
        "//nuplan/database/nuplan_db:templates",
        "//nuplan/database/utils:geometry",
    ],
)

py_library(
    name = "splitters",
    srcs = ["splitters.py"],
    deps = [
        "//nuplan/database/common:db",
        "//nuplan/database/nuplan_db:models",
        "//nuplan/database/nuplan_db:nuplandb",
    ],
)

py_library(
    name = "templates",
    srcs = ["templates.py"],
    deps = ["//nuplan/database/common:templates"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//nuplan/database/nuplan_db:frame",
        "//nuplan/database/utils:geometry",
        "//nuplan/database/utils/boxes:box3d",
        "//nuplan/database/utils/label",
        "//nuplan/database/utils/pointclouds:lidar",
    ],
)
