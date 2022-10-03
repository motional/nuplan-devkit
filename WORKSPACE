workspace(name = "nuplan_devkit")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_python",
    sha256 = "9fcf91dbcc31fde6d1edb15f117246d912c33c36f44cf681976bd886538deba6",
    strip_prefix = "rules_python-0.8.0",
    urls = [
        "https://github.com/bazelbuild/rules_python/archive/refs/tags/0.8.0.tar.gz",
    ],
)

load("@rules_python//python/pip_install:repositories.bzl", "pip_install_dependencies")

pip_install_dependencies()

load("@rules_python//python:repositories.bzl", "python_register_toolchains")

python_register_toolchains(
    name = "python3_9",
    python_version = "3.9",
)

load("@python3_9//:defs.bzl", PYTHON_INTERPRETER_TARGET = "interpreter")
load("@rules_python//python:pip.bzl", "pip_parse")

http_archive(
    name = "com_github_bazelbuild_buildtools",
    sha256 = "932160d5694e688cb7a05ac38efba4b9a90470c75f39716d85fb1d2f95eec96d",
    strip_prefix = "buildtools-4.0.1",
    urls = [
        "https://github.com/bazelbuild/buildtools/archive/refs/tags/4.0.1.zip",
    ],
)

http_archive(
    name = "bazel_skylib",
    sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
    ],
)

http_archive(
    name = "io_bazel_rules_go",
    sha256 = "685052b498b6ddfe562ca7a97736741d87916fe536623afb7da2824c0211c369",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_go/releases/download/v0.33.0/rules_go-v0.33.0.zip",
        "https://github.com/bazelbuild/rules_go/releases/download/v0.33.0/rules_go-v0.33.0.zip",
    ],
)

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

go_rules_dependencies()

go_register_toolchains(version = "1.18.3")

http_archive(
    name = "com_google_protobuf",
    sha256 = "d0f5f605d0d656007ce6c8b5a82df3037e1d8fe8b121ed42e536f569dec16113",
    strip_prefix = "protobuf-3.14.0",
    urls = [
        "https://mirror.bazel.build/github.com/protocolbuffers/protobuf/archive/v3.14.0.tar.gz",
        "https://github.com/protocolbuffers/protobuf/archive/v3.14.0.tar.gz",
    ],
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

# pip_parse() is used instead of pip_install() so that we can move pip installation to the build stage
# instead of prior to the Bazel analysis phase (which is the case for pip_install()). This change will
# help to speed up the overall build time and only download the dependencies as and when needed.
#
# pip_parse() also requires us to specify a fully resolved lock file for our Python dependencies. The
# advantage to this is that it enforces fully deterministic builds. Without fully locked dependencies,
# we cannot guarantee that two builds on the same commit use the same package versions.
#
# See:
# - https://github.com/bazelbuild/rules_python#fetch-pip-dependencies-lazily
# - https://github.com/bazelbuild/rules_python/blob/main/docs/pip.md#pip_parse
# - https://github.com/bazelbuild/rules_python/blob/main/docs/pip.md#compile_pip_requirements
#
# The helper compile_pip_requirements() is used for regenerating the locked requirements.txt files.
# Steps:
# 1) Add/remove/modify package in requirements.txt.
# 2) Validate locked requirements.txt can be generated:
#     bazel test //path/to/package_deps:package_deps_test
# 3) Update requirements_lock.txt:
#     bazel run //path/to/package_deps.update
# 4) Commit updated requirements_lock.txt

PIP_INSTALL_TIMEOUT_SECONDS = 3600  # 60 minutes

# "--only-binary=:all:" parameter below is to enforce pip to use prebuilt whl only,
# to reduce external package preparation time.
# For all packages with missing prebuilt binary, wheels are created manually and uploaded to Artifactory.
# All such packages have `+av` suffix, just in case.
# The project https://github.com/pypa/manylinux has been used - `quay.io/pypa/manylinux_2_24_x86_64` image
#
# To create and upload package, using `quay.io/pypa/manylinux_2_24_x86_64` docker image:
# - download tarball from https://pypi.org/project/
# - update `setup.py` and add `+av` suffix to version variable
# - run `python setup.py sdist bdist_wheel upload -r nutonomypip`
# Please refer to https://packaging.python.org/en/latest/tutorials/packaging-projects/ for further details

PIP_EXTRA_ARGS = [
    "--require-hashes",
    "--index-url=https://pypi.org/simple",
]

# Base Python pip dependencies
pip_parse(
    name = "pip_nuplan_devkit_deps",
    timeout = PIP_INSTALL_TIMEOUT_SECONDS,
    extra_pip_args = PIP_EXTRA_ARGS,
    python_interpreter_target = PYTHON_INTERPRETER_TARGET,
    requirements_lock = "//:requirements_lock.txt",
)

load("@pip_nuplan_devkit_deps//:requirements.bzl", install_pip_deps = "install_deps")

install_pip_deps()

pip_parse(
    name = "pip_torch_deps",
    timeout = PIP_INSTALL_TIMEOUT_SECONDS,
    extra_pip_args = PIP_EXTRA_ARGS,
    python_interpreter_target = PYTHON_INTERPRETER_TARGET,
    requirements_lock = "//:requirements_torch_lock.txt",
)

load("@pip_torch_deps//:requirements.bzl", install_pip_torch_deps = "install_deps")

install_pip_torch_deps()
