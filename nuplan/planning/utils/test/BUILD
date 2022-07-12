load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_color",
    srcs = ["test_color.py"],
    deps = [
        "//nuplan/planning/utils:color",
    ],
)

py_test(
    name = "test_scene_color",
    srcs = ["test_scene_color.py"],
    deps = [
        "//nuplan/planning/utils:color",
    ],
)
