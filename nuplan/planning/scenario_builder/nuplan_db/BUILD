load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "nuplan_scenario",
    srcs = ["nuplan_scenario.py"],
    deps = [
        ":nuplan_scenario_utils",
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/common/maps/nuplan_map:map_factory",
        "//nuplan/common/maps/nuplan_map:utils",
        "//nuplan/database/common/blob_store:local_store",
        "//nuplan/database/common/blob_store:s3_store",
        "//nuplan/database/nuplan_db:lidar_pc",
        "//nuplan/database/nuplan_db:nuplan_db_utils",
        "//nuplan/database/nuplan_db:nuplan_scenario_queries",
        "//nuplan/database/utils/pointclouds:lidar",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/scenario_builder:scenario_utils",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/trajectory:trajectory_sampling",
    ],
)

py_library(
    name = "nuplan_scenario_builder",
    srcs = ["nuplan_scenario_builder.py"],
    deps = [
        ":nuplan_scenario",
        ":nuplan_scenario_filter_utils",
        ":nuplan_scenario_utils",
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/common/maps:abstract_map_factory",
        "//nuplan/common/maps/nuplan_map:map_factory",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/scenario_builder:abstract_scenario_builder",
        "//nuplan/planning/scenario_builder:scenario_filter",
        "//nuplan/planning/utils/multithreading:worker_utils",
        requirement("pyogrio"),
    ],
)

py_library(
    name = "nuplan_scenario_filter_utils",
    srcs = ["nuplan_scenario_filter_utils.py"],
    deps = [
        ":nuplan_scenario",
        ":nuplan_scenario_utils",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/common/utils:s3_utils",
        "//nuplan/database/nuplan_db:nuplan_scenario_queries",
        "//nuplan/planning/training/preprocessing/feature_builders:vector_builder_utils",
        "//nuplan/planning/utils/multithreading:worker_utils",
        requirement("numpy"),
    ],
)

py_library(
    name = "nuplan_scenario_utils",
    srcs = ["nuplan_scenario_utils.py"],
    deps = [
        "//nuplan/common/actor_state:agent",
        "//nuplan/common/actor_state:tracked_objects",
        "//nuplan/common/actor_state:waypoint",
        "//nuplan/common/geometry:interpolate_state",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps/nuplan_map:map_factory",
        "//nuplan/database/common/blob_store:creator",
        "//nuplan/database/common/blob_store:local_store",
        "//nuplan/database/common/blob_store:s3_store",
        "//nuplan/database/nuplan_db:image",
        "//nuplan/database/nuplan_db:nuplan_db_utils",
        "//nuplan/database/nuplan_db:nuplan_scenario_queries",
        "//nuplan/database/utils:image",
        "//nuplan/database/utils/pointclouds:lidar",
        "//nuplan/planning/simulation/trajectory:predicted_trajectory",
        "//nuplan/planning/simulation/trajectory:trajectory_sampling",
    ],
)
