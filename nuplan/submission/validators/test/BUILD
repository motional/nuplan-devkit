load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_base_submission_validator",
    size = "small",
    srcs = ["test_base_submission_validator.py"],
    deps = [
        "//nuplan/submission/validators:base_submission_validator",
    ],
)

py_test(
    name = "test_image_exists_validator",
    size = "small",
    srcs = ["test_image_exists_validator.py"],
    deps = [
        "//nuplan/submission/validators:base_submission_validator",
        "//nuplan/submission/validators:image_exists_validator",
    ],
)

py_test(
    name = "test_image_is_runnable_validator",
    size = "small",
    srcs = ["test_image_is_runnable_validator.py"],
    deps = [
        "//nuplan/submission/validators:base_submission_validator",
        "//nuplan/submission/validators:image_is_runnable_validator",
    ],
)

py_test(
    name = "test_submission_computes_trajectory_validator",
    size = "small",
    srcs = ["test_submission_computes_trajectory_validator.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/submission/validators:base_submission_validator",
        "//nuplan/submission/validators:submission_computes_trajectory_validator",
    ],
)
