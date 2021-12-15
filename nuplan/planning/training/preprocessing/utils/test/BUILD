load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_agent_processing",
    size = "small",
    srcs = ["test_agent_processing.py"],
    deps = [
        "//nuplan/common/actor_state:ego_state",
        "//nuplan/common/actor_state:state_representation",
        "//nuplan/database/utils/boxes:box3d",
        "//nuplan/planning/training/preprocessing/utils:agents_preprocessing",
    ],
)
