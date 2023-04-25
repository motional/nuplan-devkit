load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "tutorial_utils",
    srcs = ["tutorial_utils.py"],
    deps = [
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/common/maps/nuplan_map:map_factory",
        "//nuplan/database/nuplan_db:nuplan_db_utils",
        "//nuplan/database/nuplan_db:nuplan_scenario_queries",
        "//nuplan/database/nuplan_db_orm:nuplandb",
        "//nuplan/database/nuplan_db_orm:nuplandb_wrapper",
        "//nuplan/planning/nuboard/base:data_class",
        "//nuplan/planning/nuboard/base:experiment_file_data",
        "//nuplan/planning/nuboard/base:simulation_tile",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_filter_utils",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_utils",
        "//nuplan/planning/simulation:simulation_log",
        "//nuplan/planning/simulation/callback:serialization_callback",
        "//nuplan/planning/simulation/controller:abstract_controller",
        "//nuplan/planning/simulation/controller:perfect_tracking",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/history:simulation_history_buffer",
        "//nuplan/planning/simulation/observation:abstract_observation",
        "//nuplan/planning/simulation/observation:tracks_observation",
        "//nuplan/planning/simulation/planner:simple_planner",
        "//nuplan/planning/simulation/simulation_time_controller:abstract_simulation_time_controller",
        "//nuplan/planning/simulation/simulation_time_controller:step_simulation_time_controller",
        "//nuplan/planning/simulation/trajectory:interpolated_trajectory",
        requirement("bokeh"),
        requirement("ipywidgets"),
        requirement("nest_asyncio"),
    ],
)
