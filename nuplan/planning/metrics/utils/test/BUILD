load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

filegroup(
    name = "test_json",
    srcs = glob(["json/**/*.json"]),
    visibility = ["//visibility:private"],
)

py_test(
    name = "test_route_extractor",
    size = "small",
    srcs = ["test_route_extractor.py"],
    data = [":test_json"],
    tags = ["integration"],
    deps = [
        "//nuplan/common/actor_state:car_footprint",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:vehicle_parameters",
        "//nuplan/common/maps/nuplan_map:map_factory",
        "//nuplan/common/utils/test_utils:nuplan_test",
        "//nuplan/database/tests:test_utils_nuplan_db",
        "//nuplan/planning/metrics/utils:route_extractor",
    ],
)
