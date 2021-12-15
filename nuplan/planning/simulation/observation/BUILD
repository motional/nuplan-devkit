load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "abstract_observation",
    srcs = ["abstract_observation.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/simulation_manager:simulation_iteration",
    ],
)

py_library(
    name = "box",
    srcs = ["box.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/observation:abstract_observation",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/simulation_manager:simulation_iteration",
    ],
)

py_library(
    name = "idm_agents_observation",
    srcs = ["idm_agents_observation.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/observation:abstract_observation",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/observation/smart_agents/idm_agents:idm_agent_manager",
        "//nuplan/planning/simulation/observation/smart_agents/idm_agents:idm_agents_builder",
        "//nuplan/planning/simulation/simulation_manager:simulation_iteration",
    ],
)

py_library(
    name = "lidar_pc",
    srcs = ["lidar_pc.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/observation:abstract_observation",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/simulation_manager:simulation_iteration",
    ],
)

py_library(
    name = "observation_type",
    srcs = ["observation_type.py"],
    deps = [
        "//nuplan/common/actor_state:tracked_objects",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/database/utils/boxes:box3d",
    ],
)
