load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_ilqr_solver",
    size = "small",
    srcs = ["test_ilqr_solver.py"],
    deps = [
        "//nuplan/planning/simulation/controller/tracker/ilqr:ilqr_solver",
    ],
)
