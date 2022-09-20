load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "idm_agent",
    srcs = ["idm_agent.py"],
    deps = [
        "//nuplan/common/actor_state:agent",
        "//nuplan/common/actor_state:oriented_box",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:tracked_objects_types",
        "//nuplan/common/actor_state:waypoint",
        "//nuplan/database/utils:geometry",
        "//nuplan/database/utils/boxes:box3d",
        "//nuplan/planning/simulation/observation/idm:idm_policy",
        "//nuplan/planning/simulation/observation/idm:idm_states",
        "//nuplan/planning/simulation/observation/idm:utils",
        "//nuplan/planning/simulation/path:interpolated_path",
        "//nuplan/planning/simulation/path:utils",
        "//nuplan/planning/simulation/trajectory:interpolated_trajectory",
    ],
)

py_library(
    name = "idm_agent_manager",
    srcs = ["idm_agent_manager.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/geometry:transform",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/planning/metrics/utils:expert_comparisons",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/observation/idm:idm_agent",
        "//nuplan/planning/simulation/observation/idm:idm_states",
        "//nuplan/planning/simulation/observation/idm:utils",
        "//nuplan/planning/simulation/occupancy_map:abstract_occupancy_map",
    ],
)

py_library(
    name = "idm_agents_builder",
    srcs = ["idm_agents_builder.py"],
    deps = [
        "//nuplan/common/actor_state:agent",
        "//nuplan/common/actor_state:oriented_box",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:tracked_objects_types",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/database/utils:geometry",
        "//nuplan/database/utils/boxes:box3d",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/observation/idm:idm_agent",
        "//nuplan/planning/simulation/observation/idm:idm_agent_manager",
        "//nuplan/planning/simulation/observation/idm:idm_policy",
        "//nuplan/planning/simulation/observation/idm:utils",
        "//nuplan/planning/simulation/occupancy_map:abstract_occupancy_map",
        "//nuplan/planning/simulation/occupancy_map:strtree_occupancy_map",
        "//nuplan/planning/simulation/path:interpolated_path",
    ],
)

py_library(
    name = "idm_policy",
    srcs = ["idm_policy.py"],
    deps = [
        "//nuplan/planning/simulation/observation/idm:idm_states",
        requirement("numpy"),
        requirement("scipy"),
    ],
)

py_library(
    name = "idm_states",
    srcs = ["idm_states.py"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:tracked_objects",
        "//nuplan/common/actor_state:transform_state",
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/common/geometry:compute",
        "//nuplan/database/utils:geometry",
        "//nuplan/database/utils/boxes:box3d",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/path:interpolated_path",
        "//nuplan/planning/simulation/path:utils",
    ],
)
