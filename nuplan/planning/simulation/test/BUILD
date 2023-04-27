load("@rules_python//python:defs.bzl", "py_test")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_simulation",
    size = "medium",
    srcs = ["test_simulation.py"],
    deps = [
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
        "//nuplan/planning/simulation",
        "//nuplan/planning/simulation:simulation_setup",
        "//nuplan/planning/simulation/callback:abstract_callback",
        "//nuplan/planning/simulation/callback:multi_callback",
        "//nuplan/planning/simulation/controller:perfect_tracking",
        "//nuplan/planning/simulation/history:simulation_history",
        "//nuplan/planning/simulation/observation:tracks_observation",
        "//nuplan/planning/simulation/runner:simulations_runner",
        "//nuplan/planning/simulation/simulation_time_controller:step_simulation_time_controller",
        requirement("hypothesis"),
    ],
)
