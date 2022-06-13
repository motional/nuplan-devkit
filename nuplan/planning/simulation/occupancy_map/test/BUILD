load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_occupancy_map",
    size = "small",
    srcs = ["test_occupancy_map.py"],
    deps = [
        "//nuplan/planning/simulation/occupancy_map:geopandas_occupancy_map",
        "//nuplan/planning/simulation/occupancy_map:strtree_occupancy_map",
    ],
)
