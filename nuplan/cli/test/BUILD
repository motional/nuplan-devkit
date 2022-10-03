load("@rules_python//python:defs.bzl", "py_test")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_nuplan_cli",
    size = "small",
    srcs = ["test_nuplan_cli.py"],
    deps = [
        "//nuplan/cli:nuplan_cli",
        "//nuplan/database/nuplan_db:db_description_types",
        requirement("typer"),
        requirement("mock"),
    ],
)
