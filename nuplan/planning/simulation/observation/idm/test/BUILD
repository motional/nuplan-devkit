load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "test_json",
    srcs = glob(["json/**/*.json"]),
    visibility = ["//visibility:private"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:tracked_objects_types",
        "//nuplan/common/maps:abstract_map_factory",
        "//nuplan/planning/simulation/observation/idm:idm_agent",
        "//nuplan/planning/simulation/observation/idm:idm_agent_manager",
        "//nuplan/planning/simulation/observation/idm:idm_agents_builder",
        "//nuplan/planning/simulation/observation/idm:idm_policy",
        "//nuplan/planning/simulation/occupancy_map:strtree_occupancy_map",
        "//nuplan/planning/utils/serialization:from_scene",
    ],
)

py_test(
    name = "test_idm_policy",
    size = "small",
    srcs = ["test_idm_policy.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/planning/simulation/observation/idm:idm_policy",
        "//nuplan/planning/simulation/observation/idm:idm_states",
        requirement("pytest"),
    ],
)

py_test(
    name = "test_utils",
    size = "small",
    srcs = ["test_utils.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/planning/simulation/observation/idm:utils",
        requirement("pytest"),
    ],
)

py_test(
    name = "test_idm_manager",
    size = "large",
    srcs = ["test_idm_manager.py"],
    data = [
        ":test_json",
    ],
    tags = ["integration"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/common/maps/nuplan_map:map_factory",
        "//nuplan/common/utils/test_utils:nuplan_test",
        "//nuplan/database/maps_db:gpkg_mapsdb",
        "//nuplan/database/tests:test_utils_nuplan_db",
        "//nuplan/planning/simulation/observation/idm:idm_policy",
        "//nuplan/planning/simulation/observation/idm/test:utils",
        "//nuplan/planning/utils/serialization:from_scene",
        requirement("pytest"),
    ],
)

py_test(
    name = "test_idm_agents_observation",
    size = "large",
    srcs = ["test_idm_agents_observation.py"],
    data = [
        ":test_json",
    ],
    tags = ["integration"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/common/maps/nuplan_map:map_factory",
        "//nuplan/common/utils/test_utils:nuplan_test",
        "//nuplan/database/nuplan_db_orm:nuplandb",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario",
        "//nuplan/planning/scenario_builder/nuplan_db/test:nuplan_scenario_test_utils",
        "//nuplan/planning/simulation/observation:idm_agents",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
        requirement("pytest"),
    ],
)

py_test(
    name = "test_profile_idm_observation",
    size = "large",
    srcs = ["test_profile_idm_observation.py"],
    data = [
    ],
    tags = ["integration"],
    deps = [
        "//nuplan/planning/scenario_builder/nuplan_db/test:nuplan_scenario_test_utils",
        "//nuplan/planning/simulation/history:simulation_history_buffer",
        "//nuplan/planning/simulation/observation:idm_agents",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
        requirement("pytest"),
        requirement("pyinstrument"),
    ],
)
