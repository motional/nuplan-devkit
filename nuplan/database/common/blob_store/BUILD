load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "blob_store",
    srcs = ["blob_store.py"],
)

py_library(
    name = "cache_store",
    srcs = ["cache_store.py"],
    deps = [
        "//nuplan/database/common/blob_store",
        "//nuplan/database/common/blob_store:local_store",
    ],
)

py_library(
    name = "creator",
    srcs = ["creator.py"],
    deps = [
        "//nuplan/database/common/blob_store",
        "//nuplan/database/common/blob_store:cache_store",
        "//nuplan/database/common/blob_store:http_store",
        "//nuplan/database/common/blob_store:local_store",
        "//nuplan/database/common/blob_store:s3_store",
        requirement("requests"),
    ],
)

py_library(
    name = "http_store",
    srcs = ["http_store.py"],
    deps = [
        "//nuplan/database/common/blob_store",
    ],
)

py_library(
    name = "local_store",
    srcs = ["local_store.py"],
    deps = ["//nuplan/database/common/blob_store"],
)

py_library(
    name = "s3_store",
    srcs = ["s3_store.py"],
    deps = [
        "//nuplan/common/utils:s3_utils",
        "//nuplan/database/common/blob_store",
        requirement("botocore"),
        requirement("tqdm"),
    ],
)
