load("@rules_python//python:defs.bzl", "py_test")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_ray_worker",
    size = "medium",
    srcs = ["test_ray_worker.py"],
    tags = ["gpu"],
    deps = [
        "//nuplan/planning/training/modeling/models:raster_model",
        "//nuplan/planning/utils/multithreading:worker_pool",
        "//nuplan/planning/utils/multithreading:worker_ray",
    ],
)

py_test(
    name = "test_splitter",
    size = "medium",
    srcs = ["test_splitter.py"],
    deps = [
        "//nuplan/planning/utils/multithreading:worker_utils",
        requirement("numpy"),
    ],
)

py_test(
    name = "test_worker_pool",
    size = "small",
    srcs = ["test_worker_pool.py"],
    tags = ["exclusive"],
    deps = [
        "//nuplan/planning/utils/multithreading:worker_parallel",
        "//nuplan/planning/utils/multithreading:worker_pool",
        "//nuplan/planning/utils/multithreading:worker_ray",
        "//nuplan/planning/utils/multithreading:worker_sequential",
        "//nuplan/planning/utils/multithreading:worker_utils",
    ],
)
