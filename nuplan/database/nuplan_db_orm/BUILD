load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

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
        "//nuplan/database/nuplan_db_orm:models",
    ],
)

py_library(
    name = "category",
    srcs = ["category.py"],
    deps = [
        "//nuplan/database/common:sql_types",
        "//nuplan/database/common:utils",
        "//nuplan/database/nuplan_db_orm:models",
        requirement("SQLAlchemy"),
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
        "//nuplan/database/nuplan_db_orm:models",
        "//nuplan/database/nuplan_db_orm:utils",
        "//nuplan/database/nuplan_db_orm:vector_map_np",
    ],
)

py_library(
    name = "lidar",
    srcs = ["lidar.py"],
    deps = [
        "//nuplan/database/common:data_types",
        "//nuplan/database/common:sql_types",
        "//nuplan/database/common:utils",
        "//nuplan/database/nuplan_db_orm:lidar_pc",
        "//nuplan/database/nuplan_db_orm:models",
    ],
)

py_library(
    name = "log",
    srcs = ["log.py"],
    deps = [
        "//nuplan/database/common:sql_types",
        "//nuplan/database/common:utils",
        "//nuplan/database/nuplan_db_orm:camera",
        "//nuplan/database/nuplan_db_orm:ego_pose",
        "//nuplan/database/nuplan_db_orm:image",
        "//nuplan/database/nuplan_db_orm:lidar",
        "//nuplan/database/nuplan_db_orm:lidar_box",
        "//nuplan/database/nuplan_db_orm:lidar_pc",
        "//nuplan/database/nuplan_db_orm:models",
        "//nuplan/database/nuplan_db_orm:scene",
        requirement("SQLAlchemy"),
    ],
)

py_library(
    name = "vector_map_np",
    srcs = ["vector_map_np.py"],
    deps = [
        requirement("pyquaternion"),
    ],
)

py_library(
    name = "image",
    srcs = ["image.py"],
    deps = [
        "//nuplan/database/common:sql_types",
        "//nuplan/database/common:utils",
        "//nuplan/database/nuplan_db_orm:camera",
        "//nuplan/database/nuplan_db_orm:ego_pose",
        "//nuplan/database/nuplan_db_orm:frame",
        "//nuplan/database/nuplan_db_orm:utils",
        "//nuplan/database/utils/boxes:box3d",
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
        "//nuplan/database/nuplan_db_orm:category",
        "//nuplan/database/nuplan_db_orm:ego_pose",
        "//nuplan/database/nuplan_db_orm:models",
        "//nuplan/database/utils:iterable_lidar_box",
        "//nuplan/database/utils/boxes:box3d",
        "//nuplan/database/utils/label:utils",
        "//nuplan/planning/simulation/trajectory:interpolated_trajectory",
        "//nuplan/planning/simulation/trajectory:predicted_trajectory",
        requirement("cachetools"),
    ],
)

py_library(
    name = "lidar_pc",
    srcs = ["lidar_pc.py"],
    deps = [
        "//nuplan/database/common:sql_types",
        "//nuplan/database/common:utils",
        "//nuplan/database/nuplan_db_orm:camera",
        "//nuplan/database/nuplan_db_orm:ego_pose",
        "//nuplan/database/nuplan_db_orm:frame",
        "//nuplan/database/nuplan_db_orm:lidar_box",
        "//nuplan/database/nuplan_db_orm:models",
        "//nuplan/database/nuplan_db_orm:scene",
        "//nuplan/database/nuplan_db_orm:utils",
        "//nuplan/database/utils/boxes:box3d",
        "//nuplan/database/utils/label:utils",
        "//nuplan/database/utils/pointclouds:lidar",
        requirement("SQLAlchemy"),
    ],
)

py_library(
    name = "models",
    srcs = ["models.py"],
    deps = [
        "//nuplan/database/common:data_types",
        "//nuplan/database/common:sql_types",
        "//nuplan/database/common:utils",
        "//nuplan/database/nuplan_db_orm:frame",
        "//nuplan/database/nuplan_db_orm:utils",
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
        "//nuplan/database/nuplan_db_orm:camera",
        "//nuplan/database/nuplan_db_orm:category",
        "//nuplan/database/nuplan_db_orm:ego_pose",
        "//nuplan/database/nuplan_db_orm:lidar_box",
        "//nuplan/database/nuplan_db_orm:lidar_pc",
        "//nuplan/database/nuplan_db_orm:log",
        "//nuplan/database/nuplan_db_orm:models",
        "//nuplan/database/nuplan_db_orm:templates",
        "//nuplan/database/nuplan_db_orm:track",
        "//nuplan/database/nuplan_db_orm:traffic_light_status",
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
        "//nuplan/database/nuplan_db_orm:nuplandb",
        "//nuplan/database/nuplan_db_orm:scene",
    ],
)

py_library(
    name = "scene",
    srcs = ["scene.py"],
    deps = [
        "//nuplan/database/common:sql_types",
        "//nuplan/database/common:utils",
        "//nuplan/database/nuplan_db_orm:ego_pose",
        "//nuplan/database/nuplan_db_orm:models",
        requirement("SQLAlchemy"),
    ],
)

py_library(
    name = "scenario_tag",
    srcs = ["scenario_tag.py"],
    deps = [
        "//nuplan/database/common:sql_types",
        "//nuplan/database/common:utils",
        "//nuplan/database/nuplan_db_orm:lidar_pc",
        "//nuplan/database/nuplan_db_orm:models",
        requirement("SQLAlchemy"),
    ],
)

py_library(
    name = "rendering_utils",
    srcs = ["rendering_utils.py"],
    deps = [
        "//nuplan/database/nuplan_db_orm:camera",
        "//nuplan/database/nuplan_db_orm:image",
        "//nuplan/database/nuplan_db_orm:lidar_box",
        "//nuplan/database/nuplan_db_orm:lidar_pc",
        "//nuplan/database/nuplan_db_orm:nuplandb",
        "//nuplan/database/utils:geometry",
        "//nuplan/database/utils/boxes:box3d",
    ],
)

py_library(
    name = "splitters",
    srcs = ["splitters.py"],
    deps = [
        "//nuplan/database/common:db",
        "//nuplan/database/nuplan_db_orm:log",
        "//nuplan/database/nuplan_db_orm:nuplandb",
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
        "//nuplan/database/nuplan_db_orm:category",
        "//nuplan/database/nuplan_db_orm:lidar_box",
        "//nuplan/database/nuplan_db_orm:lidar_pc",
        "//nuplan/database/nuplan_db_orm:scenario_tag",
    ],
)

py_library(
    name = "traffic_light_status",
    srcs = ["traffic_light_status.py"],
    deps = [
        "//nuplan/database/common:utils",
        "//nuplan/database/nuplan_db_orm:lidar_pc",
        "//nuplan/database/nuplan_db_orm:models",
        requirement("SQLAlchemy"),
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//nuplan/database/nuplan_db_orm:frame",
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
        "//nuplan/database/nuplan_db_orm:lidar_box",
        "//nuplan/database/nuplan_db_orm:lidar_pc",
        "//nuplan/planning/simulation/trajectory:interpolated_trajectory",
        "//nuplan/planning/simulation/trajectory:trajectory_sampling",
    ],
)
