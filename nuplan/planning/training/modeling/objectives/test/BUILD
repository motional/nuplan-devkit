load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_agents_imitation_objective",
    size = "small",
    srcs = ["test_agents_imitation_objective.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/planning/scenario_builder/cache:cached_scenario",
        "//nuplan/planning/training/modeling/objectives:agents_imitation_objective",
        "//nuplan/planning/training/preprocessing/features:agents_trajectories",
    ],
)

py_test(
    name = "test_trajectory_weight_decay_imitation_objective",
    size = "small",
    srcs = ["test_trajectory_weight_decay_imitation_objective.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/planning/scenario_builder/cache:cached_scenario",
        "//nuplan/planning/training/data_loader:distributed_sampler_wrapper",
        "//nuplan/planning/training/modeling/objectives:trajectory_weight_decay_imitation_objective",
        "//nuplan/planning/training/preprocessing/features:trajectory",
    ],
)

py_test(
    name = "test_imitation_objective",
    size = "small",
    srcs = ["test_imitation_objective.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/planning/scenario_builder/cache:cached_scenario",
        "//nuplan/planning/training/data_loader:distributed_sampler_wrapper",
        "//nuplan/planning/training/modeling/objectives:imitation_objective",
        "//nuplan/planning/training/preprocessing/features:trajectory",
    ],
)
