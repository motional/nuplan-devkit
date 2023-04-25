load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    deps = [
        "//nuplan/common/actor_state:agent",
        "//nuplan/common/actor_state:car_footprint",
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:ego_temporal_state",
        "//nuplan/common/actor_state:oriented_box",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:tracked_objects_types",
        "//nuplan/common/actor_state:waypoint",
        "//nuplan/planning/simulation/trajectory:predicted_trajectory",
    ],
)

py_test(
    name = "test_state_representation",
    size = "small",
    srcs = ["test_state_representation.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
    ],
)

py_test(
    name = "test_oriented_box",
    size = "small",
    srcs = ["test_oriented_box.py"],
    deps = [
        "//nuplan/common/actor_state:oriented_box",
        "//nuplan/common/actor_state:state_representation",
    ],
)

py_test(
    name = "test_scene_object",
    size = "small",
    srcs = ["test_scene_object.py"],
    deps = [
        "//nuplan/common/actor_state:scene_object",
    ],
)

py_test(
    name = "test_car_footprint",
    size = "small",
    srcs = ["test_car_footprint.py"],
    deps = [
        "//nuplan/common/actor_state:car_footprint",
        "//nuplan/common/actor_state/test:test_utils",
    ],
)

py_test(
    name = "test_ego_state",
    size = "small",
    srcs = ["test_ego_state.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state/test:test_utils",
    ],
)

py_test(
    name = "test_agent",
    size = "small",
    srcs = ["test_agent.py"],
    deps = [
        "//nuplan/common/actor_state:agent",
        "//nuplan/common/actor_state:agent_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:tracked_objects_types",
        "//nuplan/common/actor_state:waypoint",
        "//nuplan/common/actor_state/test:test_utils",
        "//nuplan/planning/simulation/trajectory:interpolated_trajectory",
    ],
)

py_test(
    name = "test_tracked_objects",
    size = "small",
    srcs = ["test_tracked_objects.py"],
    deps = [
        "//nuplan/common/actor_state:tracked_objects",
        "//nuplan/common/actor_state:tracked_objects_types",
        "//nuplan/common/actor_state/test:test_utils",
    ],
)

py_test(
    name = "test_dynamic_car_state",
    size = "small",
    srcs = ["test_dynamic_car_state.py"],
    deps = [
        "//nuplan/common/actor_state:dynamic_car_state",
        "//nuplan/common/actor_state:state_representation",
    ],
)

py_test(
    name = "test_waypoint",
    size = "small",
    srcs = ["test_waypoint.py"],
    deps = [
        "//nuplan/common/actor_state:waypoint",
    ],
)

py_test(
    name = "test_actor_temporal_state",
    size = "small",
    srcs = ["test_actor_temporal_state.py"],
    deps = [
        "//nuplan/common/actor_state:agent",
        "//nuplan/common/actor_state:agent_temporal_state",
        "//nuplan/common/actor_state:scene_object",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state:waypoint",
    ],
)
