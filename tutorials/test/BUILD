load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_nuplan_planner_tutorial_hydra",
    size = "small",
    srcs = ["test_nuplan_planner_tutorial_hydra.py"],
    data = [
        "//nuplan/planning/script:yaml",
    ],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/planning/script:run_nuboard",
        "//nuplan/planning/script:run_simulation",
        "//nuplan/planning/simulation/history:simulation_history_buffer",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/planner:simple_planner",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
        "//nuplan/planning/simulation/trajectory:abstract_trajectory",
        "//nuplan/planning/simulation/trajectory:interpolated_trajectory",
        "//tutorials/utils:tutorial_utils",
    ],
)

py_test(
    name = "test_nuplan_planner_tutorial_notebook",
    size = "small",
    srcs = ["test_nuplan_planner_tutorial_notebook.py"],
    data = [
        "//nuplan/planning/script:yaml",
    ],
    tags = ["manual"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/planning/script:run_nuboard",
        "//nuplan/planning/script:run_simulation",
        "//nuplan/planning/simulation/history:simulation_history_buffer",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/planner:simple_planner",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
        "//nuplan/planning/simulation/trajectory:abstract_trajectory",
        "//nuplan/planning/simulation/trajectory:interpolated_trajectory",
        "//tutorials/utils:tutorial_utils",
    ],
)
