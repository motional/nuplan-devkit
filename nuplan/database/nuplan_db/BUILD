load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "sensor_data_table_row",
    srcs = ["sensor_data_table_row.py"],
)

py_library(
    name = "lidar_pc",
    srcs = ["lidar_pc.py"],
    deps = [
        ":sensor_data_table_row",
    ],
)

py_library(
    name = "image",
    srcs = ["image.py"],
    deps = [
        ":sensor_data_table_row",
    ],
)

py_library(
    name = "camera",
    srcs = ["camera.py"],
    deps = [
        ":sensor_data_table_row",
    ],
)

py_library(
    name = "query_session",
    srcs = ["query_session.py"],
    deps = [
    ],
)

py_library(
    name = "nuplan_scenario_queries",
    srcs = ["nuplan_scenario_queries.py"],
    deps = [
        ":camera",
        ":image",
        ":lidar_pc",
        ":nuplan_db_utils",
        ":query_session",
        ":sensor_data_table_row",
        "//nuplan/common/actor_state:agent",
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:oriented_box",
        "//nuplan/common/actor_state:scene_object",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:static_object",
        "//nuplan/common/actor_state:tracked_objects",
        "//nuplan/common/actor_state:tracked_objects_types",
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/common/actor_state:waypoint",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/common/utils:helpers",
        "//nuplan/database/utils/label:utils",
        requirement("pyquaternion"),
    ],
)

py_library(
    name = "db_description_types",
    srcs = ["db_description_types.py"],
    deps = [],
)

py_library(
    name = "db_cli_queries",
    srcs = ["db_cli_queries.py"],
    deps = [
        ":db_description_types",
        ":query_session",
    ],
)

py_library(
    name = "nuplan_db_utils",
    srcs = ["nuplan_db_utils.py"],
)
