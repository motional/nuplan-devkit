load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "test_json",
    srcs = glob(["json/**/*.json"]),
    visibility = ["//visibility:private"],
)

py_library(
    name = "rendering_utils",
    srcs = ["rendering_utils.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/simulation/observation/smart_agents/idm_agents:idm_agent_manager",
        "//nuplan/planning/simulation/observation/smart_agents/idm_agents:utils",
    ],
)

py_test(
    name = "test_idm_policy",
    size = "small",
    srcs = ["test_idm_policy.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/planning/simulation/observation/smart_agents/idm_agents:idm_policy",
        "//nuplan/planning/simulation/observation/smart_agents/idm_agents:idm_states",
    ],
)

py_test(
    name = "test_utils",
    size = "small",
    srcs = ["test_utils.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/planning/simulation/observation/smart_agents/idm_agents:utils",
    ],
)

py_test(
    name = "test_idm_manager",
    size = "large",
    srcs = ["test_idm_manager.py"],
    data = [
        ":test_json",
    ],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/common/maps/nuplan_map:map_factory",
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/database/maps_db:gpkg_mapsdb",
        "//nuplan/database/nuplan_db:nuplandb",
        "//nuplan/database/utils/boxes:box3d",
        "//nuplan/planning/simulation/observation/smart_agents/idm_agents:idm_agent",
        "//nuplan/planning/simulation/observation/smart_agents/idm_agents:idm_agent_manager",
        "//nuplan/planning/simulation/observation/smart_agents/idm_agents:idm_agents_builder",
        "//nuplan/planning/simulation/observation/smart_agents/idm_agents:idm_policy",
        "//nuplan/planning/simulation/observation/smart_agents/occupancy_map:strtree_occupancy_map",
        "//nuplan/planning/utils/serialization:from_scene",
    ],
)

py_test(
    name = "test_idm_agents_observation",
    size = "large",
    srcs = ["test_idm_agents_observation.py"],
    data = [
        ":test_json",
    ],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/common/maps/nuplan_map:map_factory",
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/database/nuplan_db:nuplandb",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario",
        "//nuplan/planning/simulation/observation:idm_agents_observation",
        "//nuplan/planning/simulation/simulation_manager:simulation_iteration",
    ],
)

py_test(
    name = "test_profile_idm_observation",
    size = "large",
    srcs = ["test_profile_idm_observation.py"],
    data = [
    ],
    tags = ["manual"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/common/maps/nuplan_map:map_factory",
        "//nuplan/database/nuplan_db:nuplandb",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario",
        "//nuplan/planning/simulation/observation:idm_agents_observation",
        "//nuplan/planning/simulation/simulation_manager:simulation_iteration",
    ],
)
