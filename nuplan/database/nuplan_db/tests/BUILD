load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_test(
    name = "test_models",
    size = "small",
    srcs = ["test_models.py"],
    deps = [
        "//nuplan/database/nuplan_db:models",
        "//nuplan/database/nuplan_db:nuplandb",
        "//nuplan/database/utils:geometry",
    ],
)

py_test(
    name = "test_render",
    size = "small",
    srcs = ["test_render.py"],
    deps = ["//nuplan/database/nuplan_db:nuplandb"],
)

py_test(
    name = "test_utils",
    size = "medium",
    srcs = ["test_utils.py"],
    deps = [
        "//nuplan/database/nuplan_db:frame",
        "//nuplan/database/nuplan_db:nuplandb",
        "//nuplan/database/nuplan_db:utils",
        "//nuplan/database/utils/boxes:box3d",
        "//nuplan/database/utils/pointclouds:lidar",
    ],
)

py_test(
    name = "test_nuplandb",
    size = "medium",
    srcs = ["test_nuplandb.py"],
    deps = [
        "//nuplan/database/common:db",
        "//nuplan/database/nuplan_db:nuplandb",
    ],
)
