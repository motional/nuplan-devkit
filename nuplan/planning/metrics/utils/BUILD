load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

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

py_library(
    name = "expert_comparisons",
    srcs = ["expert_comparisons.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/geometry:compute",
        "//nuplan/planning/metrics/utils:state_extractors",
    ],
)

py_library(
    name = "route_extractor",
    srcs = ["route_extractor.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/common/maps:maps_datatypes",
    ],
)

py_library(
    name = "collision_utils",
    srcs = ["collision_utils.py"],
    deps = [
        "//nuplan/common/actor_state:agent",
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:scene_object",
        "//nuplan/common/actor_state:tracked_objects_types",
        "//nuplan/planning/metrics:metric_result",
    ],
)

py_test(
    name = "testing_utils",
    size = "small",
    srcs = ["testing_utils.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/common/maps/nuplan_map:map_factory",
        "//nuplan/database/maps_db:gpkg_mapsdb",
        "//nuplan/database/tests:test_utils_nuplan_db",
        "//nuplan/planning/metrics:abstract_metric",
        "//nuplan/planning/metrics:metric_result",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/history:simulation_history_buffer",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/observation/idm:utils",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/planner:simple_planner",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
        "//nuplan/planning/utils/serialization:from_scene",
    ],
)
