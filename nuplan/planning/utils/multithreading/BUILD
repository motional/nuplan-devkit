load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "ray_execution",
    srcs = ["ray_execution.py"],
    deps = [
        "//nuplan/planning/utils/multithreading:worker_pool",
    ],
)

py_library(
    name = "spliter",
    srcs = ["spliter.py"],
    deps = [
    ],
)

py_library(
    name = "worker_parallel",
    srcs = ["worker_parallel.py"],
    deps = [
        "//nuplan/planning/utils/multithreading:worker_pool",
    ],
)

py_library(
    name = "worker_pool",
    srcs = ["worker_pool.py"],
)

py_library(
    name = "worker_ray",
    srcs = ["worker_ray.py"],
    deps = [
        "//nuplan/planning/utils/multithreading:ray_execution",
        "//nuplan/planning/utils/multithreading:worker_pool",
    ],
)

py_library(
    name = "worker_sequential",
    srcs = ["worker_sequential.py"],
    deps = [
        "//nuplan/planning/utils/multithreading:worker_pool",
    ],
)
