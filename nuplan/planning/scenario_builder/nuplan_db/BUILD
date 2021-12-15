load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "nuplan_scenario",
    srcs = ["nuplan_scenario.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:tracked_objects",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/common/maps/nuplan_map:map_factory",
        "//nuplan/database/nuplan_db:models",
        "//nuplan/database/nuplan_db:nuplandb",
        "//nuplan/database/utils/boxes:box3d",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/scenario_builder:scenario_utils",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_utils",
        "//nuplan/planning/simulation/observation:observation_type",
    ],
)

py_library(
    name = "nuplan_scenario_builder",
    srcs = ["nuplan_scenario_builder.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps/nuplan_map:map_factory",
        "//nuplan/database/nuplan_db:nuplandb",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/scenario_builder:abstract_scenario_builder",
        "//nuplan/planning/scenario_builder:scenario_filter",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_utils",
        "//nuplan/planning/utils/multithreading:spliter",
        "//nuplan/planning/utils/multithreading:worker_pool",
    ],
)

py_library(
    name = "nuplan_scenario_utils",
    srcs = ["nuplan_scenario_utils.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:tracked_objects",
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/common/maps/nuplan_map:map_factory",
        "//nuplan/database/nuplan_db:models",
        "//nuplan/database/nuplan_db:nuplandb",
        "//nuplan/database/utils/boxes:box3d",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/scenario_builder:abstract_scenario_builder",
        "//nuplan/planning/scenario_builder:scenario_filter",
        "//nuplan/planning/scenario_builder:scenario_utils",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/utils/multithreading:spliter",
        "//nuplan/planning/utils/multithreading:worker_pool",
    ],
)
