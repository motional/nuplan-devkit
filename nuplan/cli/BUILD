load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "db_cli",
    srcs = ["db_cli.py"],
    deps = [
        "//nuplan/database/nuplan_db:db_cli_queries",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_utils",
    ],
)

py_binary(
    name = "nuplan_cli",
    srcs = ["nuplan_cli.py"],
    deps = [
        "//nuplan/cli:db_cli",
    ],
)
