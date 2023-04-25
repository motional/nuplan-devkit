load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_nuplan_scenario_queries",
    size = "small",
    srcs = ["test_nuplan_scenario_queries.py"],
    deps = [
        ":minimal_db_test_utils",
        "//nuplan/common/actor_state:agent",
        "//nuplan/common/actor_state:static_object",
        "//nuplan/common/actor_state:tracked_objects_types",
        "//nuplan/common/actor_state:waypoint",
        "//nuplan/common/utils:helpers",
        "//nuplan/database/nuplan_db:camera",
        "//nuplan/database/nuplan_db:image",
        "//nuplan/database/nuplan_db:lidar_pc",
        "//nuplan/database/nuplan_db:nuplan_db_utils",
        "//nuplan/database/nuplan_db:nuplan_scenario_queries",
    ],
)

py_test(
    name = "test_db_cli_queries",
    size = "small",
    srcs = ["test_db_cli_queries.py"],
    deps = [
        ":minimal_db_test_utils",
        "//nuplan/database/nuplan_db:db_cli_queries",
        "//nuplan/database/nuplan_db:db_description_types",
    ],
)

py_library(
    name = "minimal_db_test_utils",
    srcs = ["minimal_db_test_utils.py"],
    deps = [
        "//nuplan/database/common:data_types",
    ],
)

py_test(
    name = "test_nuplan_db_utils",
    size = "small",
    srcs = ["test_nuplan_db_utils.py"],
    deps = [
        "//nuplan/database/nuplan_db:nuplan_db_utils",
    ],
)
