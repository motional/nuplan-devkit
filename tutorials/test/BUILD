load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_nuplan_planner_tutorial_notebook",
    size = "medium",
    srcs = ["test_nuplan_planner_tutorial_notebook.py"],
    data = [
        "//nuplan/planning/script:yaml",
    ],
    deps = [
        "//nuplan/actor_state:ego_state",
        "//nuplan/actor_state:state_representation",
        "//nuplan/actor_state:vehicle_parameters",
        "//nuplan/maps:abstract_map",
        "//nuplan/planning/script:run_nuboard",
        "//nuplan/planning/script:run_simulation",
        "//nuplan/planning/simulation/history:simulation_history_buffer",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/simulation_manager:simulation_iteration",
        "//nuplan/planning/simulation/trajectory",
        "//nuplan/planning/simulation/trajectory:interpolated",
    ],
)
