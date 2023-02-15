load("@rules_python//python:defs.bzl", "py_test")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_lqr_tracker",
    size = "small",
    srcs = ["test_lqr_tracker.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/common/actor_state:car_footprint",
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
        "//nuplan/planning/simulation/controller/tracker:lqr",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
        "//nuplan/planning/simulation/trajectory:interpolated_trajectory",
        requirement("numpy"),
    ],
)

py_test(
    name = "test_tracker_utils",
    size = "small",
    srcs = ["test_tracker_utils.py"],
    deps = [
        "//nuplan/common/geometry:compute",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
        "//nuplan/planning/simulation/controller/tracker:tracker_utils",
        "//nuplan/planning/simulation/trajectory:interpolated_trajectory",
        requirement("numpy"),
    ],
)

py_test(
    name = "test_ilqr_tracker",
    size = "small",
    srcs = ["test_ilqr_tracker.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
        "//nuplan/planning/simulation/controller/tracker:ilqr_tracker",
        "//nuplan/planning/simulation/controller/tracker/ilqr:ilqr_solver",
        "//nuplan/planning/simulation/simulation_time_controller:simulation_iteration",
        "//nuplan/planning/simulation/trajectory:interpolated_trajectory",
        requirement("numpy"),
    ],
)
