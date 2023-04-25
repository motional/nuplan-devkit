load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "abstract_feature_builder",
    srcs = ["abstract_feature_builder.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/training/preprocessing/features:abstract_model_feature",
    ],
)

py_library(
    name = "agents_feature_builder",
    srcs = ["agents_feature_builder.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:tracked_objects",
        "//nuplan/common/actor_state:tracked_objects_types",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/scenario_builder:scenario_utils",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/trajectory:trajectory_sampling",
        "//nuplan/planning/training/preprocessing/feature_builders:scriptable_feature_builder",
        "//nuplan/planning/training/preprocessing/features:agents",
        "//nuplan/planning/training/preprocessing/features:trajectory_utils",
        "//nuplan/planning/training/preprocessing/utils:agents_preprocessing",
    ],
)

py_library(
    name = "generic_agents_feature_builder",
    srcs = ["generic_agents_feature_builder.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:tracked_objects",
        "//nuplan/common/actor_state:tracked_objects_types",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/scenario_builder:scenario_utils",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/trajectory:trajectory_sampling",
        "//nuplan/planning/training/preprocessing/feature_builders:scriptable_feature_builder",
        "//nuplan/planning/training/preprocessing/features:generic_agents",
        "//nuplan/planning/training/preprocessing/features:trajectory_utils",
        "//nuplan/planning/training/preprocessing/utils:agents_preprocessing",
    ],
)

py_library(
    name = "raster_feature_builder",
    srcs = ["raster_feature_builder.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/training/preprocessing/feature_builders:abstract_feature_builder",
        "//nuplan/planning/training/preprocessing/features:raster",
        "//nuplan/planning/training/preprocessing/features:raster_utils",
    ],
)

py_library(
    name = "scriptable_feature_builder",
    srcs = ["scriptable_feature_builder.py"],
    deps = [
        "//nuplan/planning/training/preprocessing/feature_builders:abstract_feature_builder",
    ],
)

py_library(
    name = "vector_map_feature_builder",
    srcs = ["vector_map_feature_builder.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/geometry:torch_geometry",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/training/preprocessing/feature_builders:scriptable_feature_builder",
        "//nuplan/planning/training/preprocessing/feature_builders:vector_builder_utils",
        "//nuplan/planning/training/preprocessing/features:abstract_model_feature",
        "//nuplan/planning/training/preprocessing/features:vector_map",
    ],
)

py_library(
    name = "vector_set_map_feature_builder",
    srcs = ["vector_set_map_feature_builder.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/geometry:torch_geometry",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/training/preprocessing/feature_builders:scriptable_feature_builder",
        "//nuplan/planning/training/preprocessing/feature_builders:vector_builder_utils",
        "//nuplan/planning/training/preprocessing/features:abstract_model_feature",
        "//nuplan/planning/training/preprocessing/features:vector_set_map",
        "//nuplan/planning/training/preprocessing/utils:vector_preprocessing",
    ],
)

py_library(
    name = "vector_builder_utils",
    srcs = ["vector_builder_utils.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/common/maps/nuplan_map:utils",
    ],
)
