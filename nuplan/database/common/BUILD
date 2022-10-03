load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "data_types",
    srcs = ["data_types.py"],
)

py_library(
    name = "db",
    srcs = ["db.py"],
    deps = [
        "//nuplan/database/common/blob_store",
        "//nuplan/database/common/blob_store:cache_store",
        "//nuplan/database/common/blob_store:creator",
        "//nuplan/database/maps_db:imapsdb",
        requirement("SQLAlchemy"),
    ],
)

py_library(
    name = "sql_types",
    srcs = ["sql_types.py"],
    deps = [
        "//nuplan/database/common:data_types",
    ],
)

py_library(
    name = "templates",
    srcs = ["templates.py"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
)
