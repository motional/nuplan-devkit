load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_ml_planner",
    size = "medium",
    srcs = ["test_ml_planner.py"],
    deps = [
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/simulation/history:simulation_history_buffer",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/planner/ml_planner",
        "//nuplan/planning/simulation/simulation_manager:simulation_iteration",
        "//nuplan/planning/simulation/trajectory:trajectory_sampling",
        "//nuplan/planning/training/modeling/models:raster_model",
        "//nuplan/planning/training/modeling/models:simple_vector_map_model",
        "//nuplan/planning/training/preprocessing/feature_builders:raster_feature_builder",
        "//nuplan/planning/training/preprocessing/target_builders:ego_trajectory_target_builder",
    ],
)
