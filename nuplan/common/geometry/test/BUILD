load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "test_compute",
    srcs = ["test_compute.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/geometry:compute",
    ],
)

py_library(
    name = "test_convert",
    srcs = ["test_convert.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/geometry:convert",
    ],
)

py_library(
    name = "test_transform",
    srcs = ["test_transform.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/geometry:transform",
    ],
)

filegroup(
    name = "test_json",
    srcs = glob(["json/*.json"]),
    visibility = ["//visibility:private"],
)

py_test(
    name = "test_tracked_object_interpolator",
    size = "large",
    srcs = ["test_tracked_object_interpolator.py"],
    data = [
        ":test_json",
    ],
    deps = [
        "//nuplan/common/geometry:tracked_object_interpolator",
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/utils/serialization:from_scene",
    ],
)
