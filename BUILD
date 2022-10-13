load("@com_github_bazelbuild_buildtools//buildifier:def.bzl", "buildifier")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")

package(default_visibility = ["//visibility:public"])

buildifier(
    name = "buildifier_nuplan",
    lint_mode = "fix",
    lint_warnings = ["+native-py"],
)

string_flag(
    name = "ubuntu_distro",
    build_setting_default = "focal",
    values = [
        "bionic",
        "focal",
    ],
)

config_setting(
    name = "focal",
    flag_values = {
        ":ubuntu_distro": "focal",
    },
)

filegroup(
    name = "requirements",
    data = [
        "requirements.txt",
        "requirements_torch.txt",
    ],
)

buildifier(
    name = "buildifier_test",
    lint_mode = "warn",
    lint_warnings = ["+native-py"],
    mode = "check",
)

# See the 'pip_deps' pip_parse() in WORKSPACE for further details
# Validate: bazel test <targetname>_update
# Generate: bazel run <targetname>.update
compile_pip_requirements(
    name = "pip_nuplan_devkit_deps",
    timeout = "moderate",  # Increase timeout for underlying py_tests
    extra_args = [
        "--allow-unsafe",
        "--index-url=https://pypi.org/simple",
    ],
    requirements_in = "requirements.txt",
    requirements_txt = "requirements_lock.txt",
    tags = [
        "nuplan_devkit_local",
    ],
)

compile_pip_requirements(
    name = "pip_deps_torch",
    timeout = "moderate",  # Increase timeout for underlying py_tests
    extra_args = [
        "--allow-unsafe",
        "--index-url=https://pypi.org/simple",
    ],
    requirements_in = "requirements_torch.txt",
    requirements_txt = "requirements_torch_lock.txt",
    tags = [
        "nuplan_devkit_local",
    ],
)
