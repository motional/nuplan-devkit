load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_gaussian_smooth_agent_augmentation",
    srcs = ["test_gaussian_smooth_agent_augmentation.py"],
    deps = [
        "//nuplan/planning/training/data_augmentation:gaussian_smooth_agent_augmentation",
        "//nuplan/planning/training/preprocessing/features:agents",
        "//nuplan/planning/training/preprocessing/features:trajectory",
    ],
)

py_test(
    name = "test_simple_agent_augmentation",
    srcs = ["test_simple_agent_augmentation.py"],
    deps = [
        "//nuplan/planning/training/data_augmentation:simple_agent_augmentation",
        "//nuplan/planning/training/preprocessing/features:agents",
        "//nuplan/planning/training/preprocessing/features:trajectory",
    ],
)

py_test(
    name = "test_kinematic_agent_augmentation",
    srcs = ["test_kinematic_agent_augmentation.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/planning/training/data_augmentation:kinematic_agent_augmentation",
        "//nuplan/planning/training/preprocessing/features:agents",
        "//nuplan/planning/training/preprocessing/features:trajectory",
    ],
)

py_test(
    name = "test_kinematic_history_agent_augmentation",
    srcs = ["test_kinematic_history_agent_augmentation.py"],
    deps = [
        "//nuplan/planning/training/data_augmentation:kinematic_history_agent_augmentation",
        "//nuplan/planning/training/preprocessing/features:agents",
    ],
)

py_test(
    name = "test_kinematic_history_generic_agent_augmentation",
    srcs = ["test_kinematic_history_generic_agent_augmentation.py"],
    deps = [
        "//nuplan/planning/training/data_augmentation:kinematic_history_generic_agent_augmentation",
        "//nuplan/planning/training/preprocessing/features:generic_agents",
    ],
)

py_test(
    name = "test_agent_dropout_augmentation",
    srcs = ["test_agent_dropout_augmentation.py"],
    deps = [
        "//nuplan/planning/training/data_augmentation:agent_dropout_augmentation",
        "//nuplan/planning/training/data_augmentation:data_augmentation_util",
        "//nuplan/planning/training/preprocessing/features:agents",
        "//nuplan/planning/training/preprocessing/features:trajectory",
    ],
)

py_test(
    name = "test_generic_agent_dropout_augmentation",
    srcs = ["test_generic_agent_dropout_augmentation.py"],
    deps = [
        "//nuplan/planning/training/data_augmentation:data_augmentation_util",
        "//nuplan/planning/training/data_augmentation:generic_agent_dropout_augmentation",
        "//nuplan/planning/training/preprocessing/features:generic_agents",
        "//nuplan/planning/training/preprocessing/features:trajectory",
    ],
)
