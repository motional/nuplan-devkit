load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_breadth_first_search",
    size = "small",
    srcs = ["test_breadth_first_search.py"],
    deps = [
        "@nuplan_devkit//nuplan/common/maps:abstract_map_objects",
        "@nuplan_devkit//nuplan/planning/simulation/planner/utils:breadth_first_search",
    ],
)
