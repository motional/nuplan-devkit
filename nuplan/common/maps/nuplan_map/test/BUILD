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
    name = "test_nuplan_map",
    size = "large",
    srcs = ["test_nuplan_map.py"],
    data = [":test_json"],
    tags = ["integration"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/common/maps:test_utils",
        "//nuplan/common/maps/nuplan_map",
        "//nuplan/common/maps/nuplan_map:map_factory",
        "//nuplan/common/utils/test_utils:nuplan_test",
        "//nuplan/database/maps_db:gpkg_mapsdb",
        "//nuplan/database/tests:test_utils_nuplan_db",
    ],
)

py_test(
    name = "test_baseline_path",
    size = "large",
    srcs = ["test_baseline_path.py"],
    data = [":test_json"],
    tags = ["integration"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/common/maps:test_utils",
        "//nuplan/common/maps/nuplan_map:map_factory",
        "//nuplan/common/maps/nuplan_map:polyline_map_object",
        "//nuplan/common/maps/nuplan_map:utils",
        "//nuplan/common/utils/test_utils:nuplan_test",
        "//nuplan/database/maps_db:gpkg_mapsdb",
        "//nuplan/database/tests:test_utils_nuplan_db",
    ],
)

py_test(
    name = "test_polygon_map_object",
    size = "large",
    srcs = ["test_polygon_map_object.py"],
    data = [":test_json"],
    tags = ["integration"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/common/maps:test_utils",
        "//nuplan/common/maps/nuplan_map:map_factory",
        "//nuplan/common/utils/test_utils:nuplan_test",
        "//nuplan/database/maps_db:gpkg_mapsdb",
        "//nuplan/database/tests:test_utils_nuplan_db",
    ],
)

py_test(
    name = "test_intersection",
    size = "large",
    srcs = ["test_intersection.py"],
    data = [":test_json"],
    tags = ["integration"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/common/maps:test_utils",
        "//nuplan/common/maps/nuplan_map:map_factory",
        "//nuplan/common/utils/test_utils:nuplan_test",
        "//nuplan/database/maps_db:gpkg_mapsdb",
        "//nuplan/database/tests:test_utils_nuplan_db",
    ],
)

py_test(
    name = "test_lane",
    size = "large",
    srcs = ["test_lane.py"],
    data = [":test_json"],
    tags = ["integration"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/common/maps:test_utils",
        "//nuplan/common/maps/nuplan_map:map_factory",
        "//nuplan/common/utils/test_utils:nuplan_test",
        "//nuplan/database/maps_db:gpkg_mapsdb",
        "//nuplan/database/tests:test_utils_nuplan_db",
    ],
)

py_test(
    name = "test_lane_connector",
    size = "large",
    srcs = ["test_lane_connector.py"],
    data = [":test_json"],
    tags = ["integration"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/common/maps:test_utils",
        "//nuplan/common/maps/nuplan_map:map_factory",
        "//nuplan/common/utils/test_utils:nuplan_test",
        "//nuplan/database/maps_db:gpkg_mapsdb",
        "//nuplan/database/tests:test_utils_nuplan_db",
    ],
)

py_test(
    name = "test_roadblock",
    size = "large",
    srcs = ["test_roadblock.py"],
    data = [":test_json"],
    tags = ["integration"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/common/maps:test_utils",
        "//nuplan/common/maps/nuplan_map:map_factory",
        "//nuplan/common/utils/test_utils:nuplan_test",
        "//nuplan/database/maps_db:gpkg_mapsdb",
        "//nuplan/database/tests:test_utils_nuplan_db",
    ],
)

py_test(
    name = "test_roadblock_connector",
    size = "large",
    srcs = ["test_roadblock_connector.py"],
    data = [":test_json"],
    tags = ["integration"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/common/maps:test_utils",
        "//nuplan/common/maps/nuplan_map:map_factory",
        "//nuplan/common/utils/test_utils:nuplan_test",
        "//nuplan/database/maps_db:gpkg_mapsdb",
        "//nuplan/database/tests:test_utils_nuplan_db",
    ],
)

py_test(
    name = "test_stop_line",
    size = "large",
    srcs = ["test_stop_line.py"],
    data = [":test_json"],
    tags = ["integration"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/maps:abstract_map",
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/common/maps:test_utils",
        "//nuplan/common/maps/nuplan_map:map_factory",
        "//nuplan/common/utils/test_utils:nuplan_test",
        "//nuplan/database/maps_db:gpkg_mapsdb",
        "//nuplan/database/tests:test_utils_nuplan_db",
    ],
)

py_test(
    name = "test_utils",
    size = "large",
    srcs = ["test_utils.py"],
    data = [":test_json"],
    tags = ["integration"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state/test:test_utils",
        "//nuplan/common/maps:abstract_map_objects",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/common/maps/nuplan_map:map_factory",
        "//nuplan/common/maps/nuplan_map:utils",
        "//nuplan/common/utils/test_utils:nuplan_test",
        "//nuplan/database/tests:test_utils_nuplan_db",
    ],
)
