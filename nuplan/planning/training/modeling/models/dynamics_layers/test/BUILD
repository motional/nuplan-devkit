load("@rules_python//python:defs.bzl", "py_test")
load("@pip_torch_deps//:requirements.bzl", requirement_torch = "requirement")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_kinematic_bicycle_layers_utils",
    size = "small",
    srcs = ["test_kinematic_bicycle_layers_utils.py"],
    deps = [
        "//nuplan/planning/training/modeling/models/dynamics_layers:kinematic_bicycle_layers_utils",
        "//nuplan/planning/training/preprocessing/features:agents",
    ],
)

py_test(
    name = "test_kinematic_unicycle_layers_utils",
    size = "small",
    srcs = ["test_kinematic_unicycle_layers_utils.py"],
    deps = [
        "//nuplan/planning/training/modeling/models/dynamics_layers:kinematic_unicycle_layers_utils",
        "//nuplan/planning/training/preprocessing/features:generic_agents",
    ],
)

py_test(
    name = "test_deep_dynamical_system_layer",
    size = "small",
    srcs = ["test_deep_dynamical_system_layer.py"],
    deps = [
        "//nuplan/planning/training/modeling/models/dynamics_layers:abstract_dynamics",
        "//nuplan/planning/training/modeling/models/dynamics_layers:deep_dynamical_system_layer",
        "//nuplan/planning/training/modeling/models/dynamics_layers:kinematic_bicycle_layer_rear_axle",
        "//nuplan/planning/training/modeling/models/dynamics_layers:kinematic_unicycle_layer_rear_axle",
        requirement_torch("torch"),
    ],
)
