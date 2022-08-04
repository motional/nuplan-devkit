load("@com_github_bazelbuild_buildtools//buildifier:def.bzl", "buildifier")

package(default_visibility = ["//visibility:public"])

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
