load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "abstract_observation",
    srcs = ["abstract_observation.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
    ],
)

py_library(
    name = "tracks_observation",
    srcs = ["tracks_observation.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/observation:abstract_observation",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
    ],
)

py_library(
    name = "idm_agents",
    srcs = ["idm_agents.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history_buffer",
        "//nuplan/planning/simulation/observation:abstract_observation",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/observation/idm:idm_agent_manager",
        "//nuplan/planning/simulation/observation/idm:idm_agents_builder",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
    ],
)

py_library(
    name = "lidar_pc",
    srcs = ["lidar_pc.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history_buffer",
        "//nuplan/planning/simulation/observation:abstract_observation",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
    ],
)

py_library(
    name = "observation_type",
    srcs = ["observation_type.py"],
    deps = [
        "//nuplan/common/actor_state:tracked_objects",
        "//nuplan/database/utils:image",
        "//nuplan/database/utils/boxes:box3d",
        "//nuplan/database/utils/pointclouds:lidar",
    ],
)

py_library(
    name = "abstract_ml_agents",
    srcs = ["abstract_ml_agents.py"],
    deps = [
        "//nuplan/common/actor_state:tracked_objects",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history_buffer",
        "//nuplan/planning/simulation/observation:abstract_observation",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/planner/ml_planner:model_loader",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
        "//nuplan/planning/training/modeling:torch_module_wrapper",
        "//nuplan/planning/training/modeling:types",
        "//nuplan/planning/training/preprocessing/utils:agents_preprocessing",
    ],
)

py_library(
    name = "ego_centric_ml_agents",
    srcs = ["ego_centric_ml_agents.py"],
    deps = [
        "//nuplan/common/actor_state:agent",
        "//nuplan/common/actor_state:oriented_box",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:tracked_objects",
        "//nuplan/common/actor_state:waypoint",
        "//nuplan/common/geometry:convert",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/observation:abstract_ml_agents",
        "//nuplan/planning/simulation/trajectory:interpolated_trajectory",
        "//nuplan/planning/training/modeling:torch_module_wrapper",
        "//nuplan/planning/training/modeling:types",
        "//nuplan/planning/training/preprocessing/features:agents_trajectories",
    ],
)
