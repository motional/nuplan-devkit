load("@rules_python//python:defs.bzl", "py_test")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_iterable_lidar_box",
    size = "medium",
    srcs = ["test_iterable_lidar_box.py"],
    deps = [
        "//nuplan/database/utils:iterable_lidar_box",
    ],
)

py_test(
    name = "test_geometry",
    size = "medium",
    srcs = ["test_geometry.py"],
    deps = [
        "//nuplan/database/utils:geometry",
        requirement("numpy"),
        requirement("pyquaternion"),
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
        requirement("numpy"),
        requirement("pyquaternion"),
    ],
)

py_test(
    name = "test_plot",
    size = "medium",
    srcs = ["test_plot.py"],
    deps = [
        "//nuplan/database/utils:plot",
        requirement("numpy"),
        requirement("Pillow"),
    ],
)

py_test(
    name = "test_image",
    size = "small",
    srcs = ["test_image.py"],
    deps = [
        "//nuplan/database/utils:image",
        requirement("opencv-python"),
        requirement("Pillow"),
    ],
)
