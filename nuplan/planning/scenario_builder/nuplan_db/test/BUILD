load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_nuplan_scenario",
    size = "medium",
    srcs = ["test_nuplan_scenario.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps:abstract_map_factory",
        "//nuplan/database/nuplan_db:nuplandb",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario",
    ],
)

py_test(
    name = "test_remove_invalid_goals_scenario_filter",
    srcs = ["test_remove_invalid_goals_scenario_filter.py"],
    data = [
    ],
    deps = [
        "//nuplan/planning/scenario_builder:scenario_filter",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/script/builders:worker_pool_builder",
    ],
)
