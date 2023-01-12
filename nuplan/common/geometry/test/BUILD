load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")
load("@pip_torch_deps//:requirements.bzl", requirement_torch = "requirement")

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

py_test(
    name = "test_torch_geometry",
    size = "small",
    srcs = ["test_torch_geometry.py"],
    deps = [
        "//nuplan/common/geometry:torch_geometry",
        requirement_torch("torch"),
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
    name = "test_interpolate_tracked_object",
    size = "large",
    srcs = ["test_interpolate_tracked_object.py"],
    data = [
        ":test_json",
    ],
    deps = [
        "//nuplan/common/geometry:interpolate_tracked_object",
        "//nuplan/common/utils/testing:nuplan_test",
        "//nuplan/planning/utils/serialization:from_scene",
        requirement("pytest"),
    ],
)
