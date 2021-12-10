load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//nuplan/planning/script:run_training",
    ],
)

py_test(
    name = "test_config_dataloader",
    size = "large",
    srcs = ["test_config_dataloader.py"],
    data = [
        "//nuplan/planning/script:yaml",
    ],
    deps = [
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/script/builders:model_builder",
        "//nuplan/planning/script/builders:scenario_building_builder",
        "//nuplan/planning/script/builders:training_builder",
        "//nuplan/planning/script/builders:training_metrics_builder",
        "//nuplan/planning/script/builders:worker_pool_builder",
        "//nuplan/planning/script/builders/utils:utils_config",
        "//nuplan/planning/training/data_loader:log_splitter",
        "//nuplan/planning/training/modeling/models:raster_model",
    ],
)

py_test(
    name = "test_model_build",
    size = "medium",
    srcs = ["test_model_build.py"],
    data = [
        "//nuplan/planning/script:yaml",
    ],
    deps = [
        "//nuplan/planning/script/builders:model_builder",
        "//nuplan/planning/script/builders:worker_pool_builder",
        "//nuplan/planning/script/builders/utils:utils_config",
        "//nuplan/planning/training/modeling:nn_model",
        "//nuplan/planning/training/modeling/models:lanegcn_model",
        "//nuplan/planning/training/modeling/models:raster_model",
        "//nuplan/planning/training/modeling/models:simple_vector_map_model",
        "//nuplan/planning/training/preprocessing/feature_builders:abstract_feature_builder",
        "//nuplan/planning/training/preprocessing/target_builders:abstract_target_builder",
    ],
)

py_test(
    name = "test_simulation",
    size = "medium",
    srcs = ["test_simulation.py"],
    data = [
        "//nuplan/planning/script:yaml",
    ],
    deps = [
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_acceleration",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_jerk",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/script:run_simulation",
        "//nuplan/planning/script/builders:model_builder",
        "//nuplan/planning/script/builders:worker_pool_builder",
        "//nuplan/planning/simulation/controller:log_playback",
        "//nuplan/planning/simulation/observation:box",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/planner:simple_planner",
        "//nuplan/planning/simulation/simulation_manager:step_simulation_manager",
        "//nuplan/planning/training/callbacks:profile_callback",
        "//nuplan/planning/training/modeling:planning_module",
    ],
)

py_test(
    name = "test_nuboard",
    size = "medium",
    srcs = ["test_nuboard.py"],
    data = [
        "//nuplan/planning/script:yaml",
    ],
    deps = [
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_acceleration",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_jerk",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/script:run_nuboard",
        "//nuplan/planning/script:run_simulation",
        "//nuplan/planning/script/builders:model_builder",
        "//nuplan/planning/script/builders:worker_pool_builder",
        "//nuplan/planning/simulation/controller:log_playback",
        "//nuplan/planning/simulation/controller:perfect_tracking",
        "//nuplan/planning/simulation/observation:box",
        "//nuplan/planning/simulation/observation:idm_agents_observation",
        "//nuplan/planning/simulation/observation:lidar_pc",
        "//nuplan/planning/simulation/planner:end_to_end",
        "//nuplan/planning/simulation/planner:simple_planner",
        "//nuplan/planning/simulation/simulation_manager:step_simulation_manager",
        "//nuplan/planning/training/callbacks:profile_callback",
        "//nuplan/planning/training/modeling:planning_module",
    ],
)

py_test(
    name = "test_train_raster_model",
    size = "enormous",
    srcs = ["test_train_raster_model.py"],
    data = [
        "//nuplan/planning/script:yaml",
    ],
    deps = [
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/script:run_training",
        "//nuplan/planning/script/builders:training_metrics_builder",
        "//nuplan/planning/script/builders:worker_pool_builder",
        "//nuplan/planning/script/test:utils",
        "//nuplan/planning/training/data_loader:log_splitter",
        "//nuplan/planning/training/modeling/models:raster_model",
    ],
)

py_test(
    name = "test_train_vector_model",
    size = "enormous",
    srcs = ["test_train_vector_model.py"],
    data = [
        "//nuplan/planning/script:yaml",
    ],
    deps = [
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/script:run_training",
        "//nuplan/planning/script/builders:training_metrics_builder",
        "//nuplan/planning/script/builders:worker_pool_builder",
        "//nuplan/planning/script/test:utils",
        "//nuplan/planning/training/data_loader:log_splitter",
        "//nuplan/planning/training/modeling/models:lanegcn_model",
        "//nuplan/planning/training/modeling/models:lanegcn_utils",
        "//nuplan/planning/training/modeling/models:simple_vector_map_model",
    ],
)

py_test(
    name = "test_train_overfitting",
    size = "enormous",
    srcs = ["test_train_overfitting.py"],
    data = [
        "//nuplan/planning/script:yaml",
    ],
    deps = [
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/script:run_training",
        "//nuplan/planning/script/builders:training_metrics_builder",
        "//nuplan/planning/script/builders:worker_pool_builder",
        "//nuplan/planning/script/test:utils",
        "//nuplan/planning/training/data_loader:log_splitter",
        "//nuplan/planning/training/modeling/models:raster_model",
    ],
)

py_test(
    name = "test_cache",
    size = "enormous",
    srcs = ["test_cache.py"],
    data = [
        "//nuplan/planning/script:yaml",
    ],
    deps = [
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/script:run_training",
        "//nuplan/planning/script/builders:training_metrics_builder",
        "//nuplan/planning/script/builders:worker_pool_builder",
        "//nuplan/planning/script/test:utils",
        "//nuplan/planning/training/data_loader:log_splitter",
        "//nuplan/planning/training/modeling/models:raster_model",
    ],
)
