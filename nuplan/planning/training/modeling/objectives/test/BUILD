load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_agents_imitation_objective",
    size = "small",
    srcs = ["test_agents_imitation_objective.py"],
    deps = [
        "//nuplan/planning/training/modeling/objectives:agents_imitation_objective",
        "//nuplan/planning/training/preprocessing/features:agents_trajectories",
    ],
)
