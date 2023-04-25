load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

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
    name = "test_scene_simple_trajectory",
    size = "small",
    srcs = ["test_scene_simple_trajectory.py"],
    deps = [
        "//nuplan/planning/utils/serialization:scene_simple_trajectory",
    ],
)

py_test(
    name = "test_scene_trajectory_state",
    size = "small",
    srcs = ["test_scene_trajectory_state.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/planning/utils/serialization:scene",
    ],
)

py_test(
    name = "test_scene_trajectory",
    size = "small",
    srcs = ["test_scene_trajectory.py"],
    deps = [
        "//nuplan/planning/utils:color",
        "//nuplan/planning/utils/serialization:scene",
    ],
)

py_test(
    name = "test_to_scene",
    size = "small",
    srcs = ["test_to_scene.py"],
    deps = [
        "//nuplan/planning/utils/serialization:to_scene",
    ],
)

py_test(
    name = "test_from_scene_tracked_objects",
    size = "small",
    srcs = ["test_from_scene_tracked_objects.py"],
    data = [":test_json"],
    deps = [
        "//nuplan/common/actor_state:tracked_objects_types",
        "//nuplan/common/utils/test_utils:nuplan_test",
        "//nuplan/planning/utils/serialization:from_scene",
        requirement("pytest"),
    ],
)
