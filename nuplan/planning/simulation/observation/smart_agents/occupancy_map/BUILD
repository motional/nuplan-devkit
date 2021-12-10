load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "abstract_occupancy_map",
    srcs = ["abstract_occupancy_map.py"],
)

py_library(
    name = "geopandas_occupancy_map",
    srcs = ["geopandas_occupancy_map.py"],
    deps = [
        "//nuplan/database/utils/boxes:box3d",
        "//nuplan/planning/simulation/observation/smart_agents/idm_agents:utils",
        "//nuplan/planning/simulation/observation/smart_agents/occupancy_map:abstract_occupancy_map",
    ],
)

py_library(
    name = "strtree_occupancy_map",
    srcs = ["strtree_occupancy_map.py"],
    deps = [
        "//nuplan/database/utils/boxes:box3d",
        "//nuplan/planning/simulation/observation/smart_agents/idm_agents:utils",
        "//nuplan/planning/simulation/observation/smart_agents/occupancy_map:abstract_occupancy_map",
    ],
)
