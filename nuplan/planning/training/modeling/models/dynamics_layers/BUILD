load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "abstract_dynamics",
    srcs = ["abstract_dynamics.py"],
    deps = [
    ],
)

py_library(
    name = "kinematic_bicycle_layers_utils",
    srcs = ["kinematic_bicycle_layers_utils.py"],
    deps = [
    ],
)

py_library(
    name = "kinematic_bicycle_layer_rear_axle",
    srcs = ["kinematic_bicycle_layer_rear_axle.py"],
    deps = [
        "//nuplan/planning/training/modeling/models/dynamics_layers:abstract_dynamics",
        "//nuplan/planning/training/modeling/models/dynamics_layers:kinematic_bicycle_layers_utils",
    ],
)

py_library(
    name = "kinematic_bicycle_layer_geometric_center",
    srcs = ["kinematic_bicycle_layer_geometric_center.py"],
    deps = [
        "//nuplan/planning/training/modeling/models/dynamics_layers:abstract_dynamics",
        "//nuplan/planning/training/modeling/models/dynamics_layers:kinematic_bicycle_layers_utils",
    ],
)

py_library(
    name = "kinematic_unicycle_layers_utils",
    srcs = ["kinematic_unicycle_layers_utils.py"],
    deps = [
    ],
)

py_library(
    name = "kinematic_unicycle_layer_rear_axle",
    srcs = ["kinematic_unicycle_layer_rear_axle.py"],
    deps = [
        "//nuplan/planning/training/modeling/models/dynamics_layers:abstract_dynamics",
        "//nuplan/planning/training/modeling/models/dynamics_layers:kinematic_unicycle_layers_utils",
    ],
)

py_library(
    name = "deep_dynamical_system_layer",
    srcs = ["deep_dynamical_system_layer.py"],
    deps = [
        "//nuplan/planning/training/modeling/models/dynamics_layers:abstract_dynamics",
    ],
)
