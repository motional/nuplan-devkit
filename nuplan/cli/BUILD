load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "db_cli",
    srcs = ["db_cli.py"],
    deps = [
        "//nuplan/database/nuplan_db:nuplandb",
        "//nuplan/database/nuplan_db:scenario_tag",
    ],
)

py_binary(
    name = "nuplan_cli",
    srcs = ["nuplan_cli.py"],
    deps = [
        "//nuplan/cli:db_cli",
    ],
)
