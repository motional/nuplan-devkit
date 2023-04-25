load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "nuplan_scenario_test_utils",
    srcs = ["nuplan_scenario_test_utils.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/database/common/blob_store:local_store",
        "//nuplan/database/common/blob_store:s3_store",
        "//nuplan/database/nuplan_db:image",
        "//nuplan/database/nuplan_db:lidar_pc",
        "//nuplan/database/nuplan_db:nuplan_db_utils",
        "//nuplan/database/nuplan_db:nuplan_scenario_queries",
        "//nuplan/database/tests:test_utils_nuplan_db",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_utils",
        requirement("mock"),
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
        "//nuplan/common/utils/test_utils:interface_validation",
        "//nuplan/database/common/blob_store:local_store",
        "//nuplan/database/common/blob_store:s3_store",
        "//nuplan/database/nuplan_db:image",
        "//nuplan/database/nuplan_db:sensor_data_table_row",
        "//nuplan/database/nuplan_db/test:minimal_db_test_utils",
        "//nuplan/database/utils:image",
        "//nuplan/database/utils/pointclouds:lidar",
        "//nuplan/planning/scenario_builder:abstract_scenario",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_utils",
        "//nuplan/planning/simulation/observation:observation_type",
        "//nuplan/planning/simulation/trajectory:trajectory_sampling",
        requirement("guppy3"),
        "//nuplan/planning/utils/multithreading:worker_sequential",
        requirement("mock"),
        requirement("pillow"),
        requirement("numpy"),
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
        "//nuplan/common/utils/test_utils:interface_validation",
        "//nuplan/planning/scenario_builder:abstract_scenario_builder",
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
        "//nuplan/planning/training/preprocessing/feature_builders:vector_builder_utils",
    ],
)

py_test(
    name = "test_nuplan_scenario_utils",
    size = "small",
    srcs = ["test_nuplan_scenario_utils.py"],
    data = [],
    tags = [],
    deps = [
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_utils",
    ],
)

py_test(
    name = "test_nuplan_scenario_utils_integration",
    size = "small",
    srcs = ["test_nuplan_scenario_utils_integration.py"],
    data = [],
    tags = ["manual"],
    deps = [
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_utils",
    ],
)

py_test(
    name = "test_nuplan_scenario_integration",
    size = "small",
    srcs = ["test_nuplan_scenario_integration.py"],
    data = [],
    tags = ["manual"],
    deps = [
        "//nuplan/planning/scenario_builder/nuplan_db/test:nuplan_scenario_test_utils",
        "//nuplan/planning/simulation/observation:observation_type",
    ],
)
