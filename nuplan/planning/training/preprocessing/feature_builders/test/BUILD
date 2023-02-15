load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_agents_feature_builder",
    size = "medium",
    srcs = ["test_agents_feature_builder.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
        "//nuplan/planning/training/preprocessing/feature_builders:agents_feature_builder",
        "//nuplan/planning/training/preprocessing/features:agents",
        "//nuplan/planning/training/preprocessing/utils:agents_preprocessing",
    ],
)

py_test(
    name = "test_generic_agents_feature_builder",
    size = "medium",
    srcs = ["test_generic_agents_feature_builder.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
        "//nuplan/planning/training/preprocessing/feature_builders:generic_agents_feature_builder",
        "//nuplan/planning/training/preprocessing/features:generic_agents",
        "//nuplan/planning/training/preprocessing/utils:agents_preprocessing",
    ],
)

py_test(
    name = "test_vector_building",
    size = "medium",
    srcs = ["test_vector_building.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
        "//nuplan/planning/training/preprocessing/feature_builders:vector_builder_utils",
    ],
)

py_test(
    name = "test_vector_map_feature_builder",
    size = "medium",
    srcs = ["test_vector_map_feature_builder.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_utils",
        "//nuplan/planning/scenario_builder/nuplan_db/test:nuplan_scenario_test_utils",
        "//nuplan/planning/training/preprocessing/feature_builders:vector_map_feature_builder",
        "//nuplan/planning/training/preprocessing/features:vector_map",
    ],
)

py_test(
    name = "test_vector_set_map_feature_builder",
    size = "medium",
    srcs = ["test_vector_set_map_feature_builder.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
        "//nuplan/planning/simulation/history:simulation_history_buffer",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
        "//nuplan/planning/training/preprocessing/feature_builders:vector_set_map_feature_builder",
        "//nuplan/planning/training/preprocessing/features:vector_set_map",
    ],
)
