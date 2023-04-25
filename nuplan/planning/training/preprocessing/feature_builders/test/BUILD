load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")
load("@pip_torch_deps//:requirements.bzl", requirement_torch = "requirement")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_agents_feature_builder",
    size = "medium",
    srcs = ["test_agents_feature_builder.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:tracked_objects_types",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
        "//nuplan/planning/training/preprocessing/feature_builders:agents_feature_builder",
        "//nuplan/planning/training/preprocessing/features:agents",
        "//nuplan/planning/training/preprocessing/utils:agents_preprocessing",
        requirement("hypothesis"),
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
    name = "test_vector_map_feature_builder_lane_metadata",
    size = "medium",
    srcs = ["test_vector_map_feature_builder_lane_metadata.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state/test:test_utils",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/common/utils/test_utils:function_validation",
        "//nuplan/common/utils/test_utils:patch",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/training/preprocessing/feature_builders:vector_builder_utils",
        "//nuplan/planning/training/preprocessing/feature_builders:vector_map_feature_builder",
        "//nuplan/planning/training/preprocessing/features:vector_map",
        requirement("numpy"),
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

py_library(
    name = "mock_feature_utils",
    srcs = ["mock_feature_utils.py"],
    deps = [
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/trajectory:trajectory_sampling",
        "//nuplan/planning/training/modeling:torch_module_wrapper",
        "//nuplan/planning/training/modeling:types",
        "//nuplan/planning/training/preprocessing/feature_builders:abstract_feature_builder",
        "//nuplan/planning/training/preprocessing/features:abstract_model_feature",
        "//nuplan/planning/training/preprocessing/features:trajectory",
        "//nuplan/planning/training/preprocessing/target_builders:abstract_target_builder",
        requirement_torch("torch"),
    ],
)
