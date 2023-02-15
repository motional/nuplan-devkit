load("@rules_python//python:defs.bzl", "py_library")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")
load("@pip_torch_deps//:requirements.bzl", requirement_torch = "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "ray_execution",
    srcs = ["ray_execution.py"],
    deps = [
        "//nuplan/planning/utils/multithreading:worker_pool",
        requirement("ray"),
        requirement("tqdm"),
    ],
)

py_library(
    name = "worker_utils",
    srcs = ["worker_utils.py"],
    deps = [
        "//nuplan/planning/utils/multithreading:worker_pool",
        requirement("numpy"),
        requirement("psutil"),
    ],
)

py_library(
    name = "worker_parallel",
    srcs = ["worker_parallel.py"],
    deps = [
        "//nuplan/planning/utils/multithreading:worker_pool",
        requirement("tqdm"),
    ],
)

py_library(
    name = "worker_pool",
    srcs = ["worker_pool.py"],
    deps = [
        requirement("psutil"),
    ],
)

py_library(
    name = "worker_ray",
    srcs = ["worker_ray.py"],
    deps = [
        "//nuplan/planning/utils/multithreading:ray_execution",
        "//nuplan/planning/utils/multithreading:worker_pool",
        requirement("ray"),
        requirement_torch("torch"),
        requirement("psutil"),
    ],
)

py_library(
    name = "worker_sequential",
    srcs = ["worker_sequential.py"],
    deps = [
        "//nuplan/planning/utils/multithreading:worker_pool",
        requirement("tqdm"),
    ],
)
