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
    name = "camera",
    srcs = ["camera.py"],
    deps = [
        "//nuplan/database/common:data_types",
        "//nuplan/database/common:sql_types",
        "//nuplan/database/common:utils",
        "//nuplan/database/nuplan_db:models",
    ],
)

py_library(
    name = "category",
    srcs = ["category.py"],
    deps = [
        "//nuplan/database/common:sql_types",
        "//nuplan/database/common:utils",
        "//nuplan/database/nuplan_db:models",
    ],
)

py_library(
    name = "ego_pose",
    srcs = ["ego_pose.py"],
    deps = [
        "//nuplan/database/common:sql_types",
        "//nuplan/database/common:utils",
        "//nuplan/database/maps_db:gpkg_mapsdb",
        "//nuplan/database/maps_db:utils",
        "//nuplan/database/nuplan_db:models",
        "//nuplan/database/nuplan_db:utils",
        "//nuplan/database/nuplan_db:vector_map_np",
    ],
)

py_library(
    name = "lidar",
    srcs = ["lidar.py"],
    deps = [
        "//nuplan/database/common:data_types",
        "//nuplan/database/common:sql_types",
        "//nuplan/database/common:utils",
        "//nuplan/database/nuplan_db:lidar_pc",
        "//nuplan/database/nuplan_db:models",
    ],
)

py_library(
    name = "log",
    srcs = ["log.py"],
    deps = [
        "//nuplan/database/common:sql_types",
        "//nuplan/database/common:utils",
        "//nuplan/database/nuplan_db:camera",
        "//nuplan/database/nuplan_db:ego_pose",
        "//nuplan/database/nuplan_db:lidar",
        "//nuplan/database/nuplan_db:lidar_box",
        "//nuplan/database/nuplan_db:lidar_pc",
        "//nuplan/database/nuplan_db:models",
        "//nuplan/database/nuplan_db:scene",
    ],
)

py_library(
    name = "vector_map_np",
    srcs = ["vector_map_np.py"],
    deps = [
    ],
)

py_library(
    name = "lidar_box",
    srcs = ["lidar_box.py"],
    deps = [
        "//nuplan/common/actor_state:agent",
        "//nuplan/common/actor_state:oriented_box",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:static_object",
        "//nuplan/common/actor_state:tracked_objects",
        "//nuplan/common/actor_state:tracked_objects_types",
        "//nuplan/common/actor_state:waypoint",
        "//nuplan/database/common:sql_types",
        "//nuplan/database/common:utils",
        "//nuplan/database/maps_db:utils",
        "//nuplan/database/nuplan_db:category",
        "//nuplan/database/nuplan_db:ego_pose",
        "//nuplan/database/nuplan_db:models",
        "//nuplan/database/utils/boxes:box3d",
        "//nuplan/database/utils/label:utils",
        "//nuplan/planning/simulation/trajectory:interpolated_trajectory",
        "//nuplan/planning/simulation/trajectory:predicted_trajectory",
    ],
)

py_library(
    name = "lidar_pc",
    srcs = ["lidar_pc.py"],
    deps = [
        "//nuplan/database/common:sql_types",
        "//nuplan/database/common:utils",
        "//nuplan/database/nuplan_db:camera",
        "//nuplan/database/nuplan_db:ego_pose",
        "//nuplan/database/nuplan_db:frame",
        "//nuplan/database/nuplan_db:lidar_box",
        "//nuplan/database/nuplan_db:models",
        "//nuplan/database/nuplan_db:scene",
        "//nuplan/database/nuplan_db:utils",
        "//nuplan/database/utils/boxes:box3d",
        "//nuplan/database/utils/label:utils",
        "//nuplan/database/utils/pointclouds:lidar",
    ],
)

py_library(
    name = "models",
    srcs = ["models.py"],
    deps = [
        "//nuplan/database/common:data_types",
        "//nuplan/database/common:sql_types",
        "//nuplan/database/common:utils",
        "//nuplan/database/nuplan_db:frame",
        "//nuplan/database/nuplan_db:utils",
        "//nuplan/database/utils/boxes:box3d",
        "//nuplan/database/utils/label:utils",
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
        "//nuplan/database/nuplan_db:camera",
        "//nuplan/database/nuplan_db:category",
        "//nuplan/database/nuplan_db:ego_pose",
        "//nuplan/database/nuplan_db:lidar_box",
        "//nuplan/database/nuplan_db:lidar_pc",
        "//nuplan/database/nuplan_db:log",
        "//nuplan/database/nuplan_db:models",
        "//nuplan/database/nuplan_db:templates",
        "//nuplan/database/nuplan_db:track",
        "//nuplan/database/nuplan_db:traffic_light_status",
        "//nuplan/database/utils:geometry",
    ],
)

py_library(
    name = "nuplandb_wrapper",
    srcs = ["nuplandb_wrapper.py"],
    deps = [
        "//nuplan/common/utils:s3_utils",
        "//nuplan/database/common/blob_store:s3_store",
        "//nuplan/database/maps_db:gpkg_mapsdb",
        "//nuplan/database/nuplan_db:nuplandb",
        "//nuplan/database/nuplan_db:scene",
    ],
)

py_library(
    name = "scene",
    srcs = ["scene.py"],
    deps = [
        "//nuplan/database/common:sql_types",
        "//nuplan/database/common:utils",
        "//nuplan/database/nuplan_db:ego_pose",
        "//nuplan/database/nuplan_db:models",
    ],
)

py_library(
    name = "scenario_tag",
    srcs = ["scenario_tag.py"],
    deps = [
        "//nuplan/database/common:sql_types",
        "//nuplan/database/common:utils",
        "//nuplan/database/nuplan_db:lidar_pc",
        "//nuplan/database/nuplan_db:models",
    ],
)

py_library(
    name = "rendering_utils",
    srcs = ["rendering_utils.py"],
    deps = [
        "//nuplan/database/nuplan_db:camera",
        "//nuplan/database/nuplan_db:lidar_box",
        "//nuplan/database/nuplan_db:lidar_pc",
        "//nuplan/database/nuplan_db:models",
        "//nuplan/database/nuplan_db:nuplandb",
        "//nuplan/database/utils:geometry",
        "//nuplan/database/utils/boxes:box3d",
    ],
)

py_library(
    name = "splitters",
    srcs = ["splitters.py"],
    deps = [
        "//nuplan/database/common:db",
        "//nuplan/database/nuplan_db:log",
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
    name = "track",
    srcs = ["track.py"],
    deps = [
        "//nuplan/database/common:sql_types",
        "//nuplan/database/nuplan_db:category",
        "//nuplan/database/nuplan_db:lidar_box",
        "//nuplan/database/nuplan_db:lidar_pc",
        "//nuplan/database/nuplan_db:scenario_tag",
    ],
)

py_library(
    name = "traffic_light_status",
    srcs = ["traffic_light_status.py"],
    deps = [
        "//nuplan/database/common:utils",
        "//nuplan/database/nuplan_db:lidar_pc",
        "//nuplan/database/nuplan_db:models",
    ],
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

py_library(
    name = "prediction_construction",
    srcs = ["prediction_construction.py"],
    deps = [
        "//nuplan/common/actor_state:oriented_box",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:waypoint",
        "//nuplan/database/nuplan_db:lidar_box",
        "//nuplan/database/nuplan_db:lidar_pc",
        "//nuplan/database/nuplan_db:models",
        "//nuplan/planning/simulation/trajectory:interpolated_trajectory",
        "//nuplan/planning/simulation/trajectory:trajectory_sampling",
    ],
)
