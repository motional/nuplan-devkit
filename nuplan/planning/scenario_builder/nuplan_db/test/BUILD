load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "nuplan_scenario_test_utils",
    srcs = ["nuplan_scenario_test_utils.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/database/nuplan_db:nuplandb",
        "//nuplan/database/tests:nuplan_db_test_utils",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_utils",
    ],
)

py_test(
    name = "test_nuplan_scenario",
    size = "medium",
    srcs = ["test_nuplan_scenario.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps:abstract_map_factory",
        "//nuplan/database/nuplan_db:nuplandb",
        "//nuplan/database/tests:nuplan_db_test_utils",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/simulation/trajectory:trajectory_sampling",
    ],
)

py_test(
    name = "test_nuplan_scenario_builder",
    size = "medium",
    srcs = ["test_nuplan_scenario_builder.py"],
    data = [
    ],
    deps = [
        "//nuplan/planning/scenario_builder:scenario_filter",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/scenario_builder/nuplan_db/test:nuplan_scenario_test_utils",
        "//nuplan/planning/script/builders:worker_pool_builder",
    ],
)
