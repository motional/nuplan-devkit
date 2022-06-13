load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "mock_s3_store",
    srcs = ["mock_s3_store.py"],
    deps = [
        "//nuplan/database/common/blob_store",
    ],
)
