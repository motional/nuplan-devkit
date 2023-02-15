load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "nuplan_scenario_test_utils",
    srcs = ["nuplan_scenario_test_utils.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/database/nuplan_db:nuplan_scenario_queries",
        "//nuplan/database/tests:nuplan_db_test_utils",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_utils",
    ],
)

py_test(
    name = "test_nuplan_scenario",
    size = "small",
    srcs = ["test_nuplan_scenario.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/common/actor_state:agent",
        "//nuplan/common/actor_state:oriented_box",
        "//nuplan/common/actor_state:scene_object",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:static_object",
        "//nuplan/common/actor_state:tracked_objects",
        "//nuplan/common/actor_state:tracked_objects_types",
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/common/actor_state:waypoint",
        "//nuplan/common/utils:interpolatable_state",
        "//nuplan/database/nuplan_db/test:minimal_db_test_utils",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_utils",
        "//nuplan/planning/simulation/trajectory:trajectory_sampling",
        requirement("guppy3"),
        "//nuplan/planning/utils/multithreading:worker_sequential",
        requirement("mock"),
    ],
)

py_test(
    name = "test_nuplan_scenario_builder",
    size = "small",
    srcs = ["test_nuplan_scenario_builder.py"],
    data = [
    ],
    tags = ["integration"],
    deps = [
        "//nuplan/planning/scenario_builder:scenario_filter",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_filter_utils",
        "//nuplan/planning/utils/multithreading:worker_sequential",
        requirement("mock"),
    ],
)

py_test(
    name = "test_nuplan_scenario_filter_utils",
    size = "small",
    srcs = ["test_nuplan_scenario_filter_utils.py"],
    data = [
    ],
    tags = [],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/scenario_builder/cache:cached_scenario",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_filter_utils",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
    ],
)
