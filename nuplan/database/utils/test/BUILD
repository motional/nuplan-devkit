load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_geometry",
    size = "medium",
    srcs = ["test_geometry.py"],
    deps = [
        "//nuplan/database/utils:geometry",
    ],
)

py_test(
    name = "test_measure",
    size = "medium",
    srcs = ["test_measure.py"],
    deps = [
        "//nuplan/database/utils:geometry",
        "//nuplan/database/utils:measure",
        "//nuplan/database/utils/boxes:box3d",
    ],
)

py_test(
    name = "test_plot",
    size = "medium",
    srcs = ["test_plot.py"],
    deps = [
        "//nuplan/database/utils:plot",
    ],
)
