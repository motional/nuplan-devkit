load("@rules_python//python:defs.bzl", "py_test")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_submission_container",
    size = "small",
    srcs = ["test_submission_container.py"],
    deps = [
        "//nuplan/submission:submission_container",
        requirement("docker"),
    ],
)

py_test(
    name = "test_challenge_servicers",
    size = "small",
    srcs = ["test_challenge_servicers.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/submission:challenge_pb2",
        "//nuplan/submission:challenge_servicers",
    ],
)

py_test(
    name = "test_submission_planner",
    size = "small",
    srcs = ["test_submission_planner.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/submission:submission_planner",
    ],
)

py_test(
    name = "test_submission_container_factory",
    size = "small",
    srcs = ["test_submission_container_factory.py"],
    deps = [
        "//nuplan/submission:submission_container_factory",
    ],
)

py_test(
    name = "test_utils",
    size = "small",
    srcs = ["test_utils.py"],
    deps = [
        "//nuplan/submission/utils",
    ],
)

py_test(
    name = "test_proto_converters",
    size = "small",
    srcs = ["test_proto_converters.py"],
    deps = [
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/common/actor_state/test:test_utils",
        "//nuplan/common/maps:maps_datatypes",
        "//nuplan/planning/simulation/trajectory:interpolated_trajectory",
        "//nuplan/submission:proto_converters",
    ],
)
