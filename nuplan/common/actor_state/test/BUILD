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
        "//nuplan/common/actor_state:oriented_box",
        "//nuplan/common/actor_state:state_representation",
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
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state/test:test_utils",
    ],
)

py_test(
    name = "test_agent",
    size = "small",
    srcs = ["test_agent.py"],
    deps = [
        "//nuplan/common/actor_state:agent",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state/test:test_utils",
    ],
)

py_test(
    name = "test_tracked_objects",
    size = "small",
    srcs = ["test_tracked_objects.py"],
    deps = [
        "//nuplan/common/actor_state:agent",
        "//nuplan/common/actor_state:tracked_objects",
        "//nuplan/common/actor_state/test:test_utils",
    ],
)
