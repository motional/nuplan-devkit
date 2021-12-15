load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "geometry",
    srcs = ["geometry.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:transform_state",
        "//nuplan/common/actor_state:vehicle_parameters",
    ],
)

py_library(
    name = "metric_violation_aggregator",
    srcs = ["metric_violation_aggregator.py"],
    deps = [
        "//nuplan/planning/metrics:metric_result",
    ],
)

py_library(
    name = "state_extractors",
    srcs = ["state_extractors.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:transform_state",
        "//nuplan/database/utils/boxes:box3d",
        "//nuplan/planning/simulation/history:simulation_history",
    ],
)

py_test(
    name = "testing_utils",
    size = "small",
    srcs = ["testing_utils.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/common/maps/nuplan_map:map_factory",
        "//nuplan/database/maps_db:gpkg_mapsdb",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/observation/smart_agents/idm_agents:utils",
        "//nuplan/planning/simulation/simulation_manager:simulation_iteration",
        "//nuplan/planning/simulation/trajectory:interpolated",
        "//nuplan/planning/utils/serialization:from_scene",
    ],
)
