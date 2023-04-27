load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@pip_torch_deps//:requirements.bzl", requirement_torch = "requirement")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "skeleton_test_train",
    srcs = ["skeleton_test_train.py"],
    data = [
        "//nuplan/planning/script:run_training",
    ],
)

py_library(
    name = "skeleton_test_simulation",
    srcs = ["skeleton_test_simulation.py"],
    data = [
        "//nuplan/planning/script:run_simulation",
    ],
    deps = [
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario",
        "//nuplan/planning/script/builders:lr_scheduler_builder",
        "//nuplan/planning/script/builders:model_builder",
        "//nuplan/planning/script/builders:worker_pool_builder",
        "//nuplan/planning/simulation/controller:log_playback",
        "//nuplan/planning/simulation/observation:tracks_observation",
        "//nuplan/planning/simulation/planner:abstract_planner",
        "//nuplan/planning/simulation/planner:simple_planner",
        "//nuplan/planning/simulation/simulation_time_controller:step_simulation_time_controller",
        "//nuplan/planning/training/callbacks:profile_callback",
        "//nuplan/planning/training/modeling:lightning_module_wrapper",
    ],
)

py_test(
    name = "test_config_dataloader",
    size = "large",
    srcs = ["test_config_dataloader.py"],
    data = [
        "//nuplan/planning/script:init",
        "//nuplan/planning/script:yaml",
    ],
    tags = ["integration"],
    deps = [
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/script/builders:data_augmentation_builder",
        "//nuplan/planning/script/builders:lr_scheduler_builder",
        "//nuplan/planning/script/builders:model_builder",
        "//nuplan/planning/script/builders:scenario_building_builder",
        "//nuplan/planning/script/builders:training_builder",
        "//nuplan/planning/script/builders:training_callback_builder",
        "//nuplan/planning/script/builders:training_metrics_builder",
        "//nuplan/planning/script/builders:worker_pool_builder",
        "//nuplan/planning/script/builders/utils:utils_config",
        "//nuplan/planning/training/data_augmentation:kinematic_agent_augmentation",
        "//nuplan/planning/training/data_loader:distributed_sampler_wrapper",
        "//nuplan/planning/training/data_loader:log_splitter",
        "//nuplan/planning/training/experiments:cache_metadata_entry",
        "//nuplan/planning/training/modeling/models:raster_model",
        "//nuplan/planning/utils/multithreading:worker_pool",
        requirement_torch("pytorch-lightning"),
    ],
)

py_test(
    name = "test_model_build",
    size = "medium",
    srcs = ["test_model_build.py"],
    data = [
        "//nuplan/planning/script:init",
        "//nuplan/planning/script:yaml",
    ],
    tags = ["integration"],
    deps = [
        "//nuplan/planning/script/builders:model_builder",
        "//nuplan/planning/script/builders:worker_pool_builder",
        "//nuplan/planning/script/builders/utils:utils_config",
        "//nuplan/planning/training/modeling:torch_module_wrapper",
        "//nuplan/planning/training/modeling/models:lanegcn_model",
        "//nuplan/planning/training/modeling/models:raster_model",
        "//nuplan/planning/training/modeling/models:simple_vector_map_model",
        "//nuplan/planning/training/modeling/models:urban_driver_open_loop_model",
        "//nuplan/planning/training/preprocessing/feature_builders:abstract_feature_builder",
        "//nuplan/planning/training/preprocessing/target_builders:abstract_target_builder",
        requirement("pyquaternion"),
    ],
)

py_test(
    name = "test_run_simulation",
    size = "small",
    srcs = ["test_run_simulation.py"],
    tags = ["integration"],
    deps = [
        ":skeleton_test_simulation",
    ],
)

py_test(
    name = "test_run_submission_planner",
    size = "small",
    srcs = ["test_run_submission_planner.py"],
    data = [
        "//nuplan/planning/script:yaml",
    ],
    tags = ["integration"],
    deps = [
        ":skeleton_test_simulation",
        "//nuplan/planning/script:run_submission_planner",
        requirement("protobuf"),
    ],
)

py_test(
    name = "test_run_challenge",
    size = "small",
    srcs = ["test_run_challenge.py"],
    tags = ["integration"],
    deps = [
        ":skeleton_test_simulation",
    ],
)

py_test(
    name = "test_run_ray_worker",
    size = "small",
    srcs = ["test_run_ray_worker.py"],
    tags = ["integration"],
    deps = [
        ":skeleton_test_simulation",
    ],
)

py_test(
    name = "test_run_result_processor",
    size = "small",
    srcs = ["test_run_result_processor.py"],
    data = [
        "//nuplan/planning/script:run_result_processor_aws",
    ],
    tags = [],
    deps = [],
)

py_test(
    name = "test_run_parallel_worker",
    size = "small",
    srcs = ["test_run_parallel_worker.py"],
    tags = ["integration"],
    deps = [
        ":skeleton_test_simulation",
    ],
)

py_test(
    name = "test_run_sequential_worker",
    size = "small",
    srcs = ["test_run_sequential_worker.py"],
    tags = ["integration"],
    deps = [
        ":skeleton_test_simulation",
    ],
)

py_test(
    name = "test_run_metric",
    size = "small",
    srcs = ["test_run_metric.py"],
    data = [
        "//nuplan/planning/script:run_metric",
        "//nuplan/planning/script:run_metric_aggregator",
    ],
    tags = ["integration"],
    deps = [
        ":skeleton_test_simulation",
    ],
)

py_test(
    name = "test_run_metric_aggregator",
    size = "small",
    srcs = ["test_run_metric_aggregator.py"],
    data = [
        "//nuplan/planning/script:run_metric",
        "//nuplan/planning/script:run_metric_aggregator",
    ],
    tags = ["integration"],
    deps = [
        ":skeleton_test_simulation",
        requirement("hydra-core"),
    ],
)

py_test(
    name = "test_run_nuboard",
    size = "medium",
    srcs = ["test_run_nuboard.py"],
    data = [
        "//nuplan/planning/script:run_nuboard",
        "//nuplan/planning/script:run_simulation",
    ],
    tags = ["integration"],
    deps = [
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_acceleration",
        "//nuplan/planning/metrics/evaluation_metrics/common:ego_jerk",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/script/builders:lr_scheduler_builder",
        "//nuplan/planning/script/builders:model_builder",
        "//nuplan/planning/script/builders:worker_pool_builder",
        "//nuplan/planning/simulation/controller:log_playback",
        "//nuplan/planning/simulation/controller:perfect_tracking",
        "//nuplan/planning/simulation/observation:idm_agents",
        "//nuplan/planning/simulation/observation:lidar_pc",
        "//nuplan/planning/simulation/observation:tracks_observation",
        "//nuplan/planning/simulation/planner:simple_planner",
        "//nuplan/planning/simulation/simulation_time_controller:step_simulation_time_controller",
        "//nuplan/planning/training/callbacks:profile_callback",
        "//nuplan/planning/training/modeling:lightning_module_wrapper",
    ],
)

py_test(
    name = "test_train_raster_model",
    size = "enormous",
    srcs = ["test_train_raster_model.py"],
    tags = ["manual"],
    deps = [
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/script/builders:data_augmentation_builder",
        "//nuplan/planning/script/builders:training_callback_builder",
        "//nuplan/planning/script/builders:training_metrics_builder",
        "//nuplan/planning/script/builders:worker_pool_builder",
        "//nuplan/planning/script/test:skeleton_test_train",
        "//nuplan/planning/training/callbacks:scenario_scoring_callback",
        "//nuplan/planning/training/callbacks/utils:scenario_scene_converter",
        "//nuplan/planning/training/data_augmentation:kinematic_agent_augmentation",
        "//nuplan/planning/training/data_loader:log_splitter",
        "//nuplan/planning/training/modeling/models:raster_model",
    ],
)

py_test(
    name = "test_train_urban_driver_open_loop_model",
    size = "enormous",
    srcs = ["test_train_urban_driver_open_loop_model.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/script/builders:data_augmentation_builder",
        "//nuplan/planning/script/builders:lr_scheduler_builder",
        "//nuplan/planning/script/builders:training_callback_builder",
        "//nuplan/planning/script/builders:training_metrics_builder",
        "//nuplan/planning/script/builders:worker_pool_builder",
        "//nuplan/planning/script/test:skeleton_test_train",
        "//nuplan/planning/training/callbacks:scenario_scoring_callback",
        "//nuplan/planning/training/callbacks:stepwise_augmentation_scheduler",
        "//nuplan/planning/training/callbacks/utils:scenario_scene_converter",
        "//nuplan/planning/training/data_augmentation:data_augmentation_util",
        "//nuplan/planning/training/data_augmentation:generic_agent_dropout_augmentation",
        "//nuplan/planning/training/data_augmentation:kinematic_history_generic_agent_augmentation",
        "//nuplan/planning/training/data_loader:distributed_sampler_wrapper",
        "//nuplan/planning/training/data_loader:log_splitter",
        "//nuplan/planning/training/experiments:cache_metadata_entry",
        "//nuplan/planning/training/modeling/models:urban_driver_open_loop_model",
        "//nuplan/planning/training/modeling/models:urban_driver_open_loop_model_utils",
        "//nuplan/planning/training/modeling/objectives:trajectory_weight_decay_imitation_objective",
        "//nuplan/planning/utils/multithreading:worker_pool",
    ],
)

py_test(
    name = "test_train_vector_model",
    size = "enormous",
    srcs = ["test_train_vector_model.py"],
    tags = ["manual"],
    deps = [
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/script/builders:data_augmentation_builder",
        "//nuplan/planning/script/builders:training_callback_builder",
        "//nuplan/planning/script/builders:training_metrics_builder",
        "//nuplan/planning/script/builders:worker_pool_builder",
        "//nuplan/planning/script/test:skeleton_test_train",
        "//nuplan/planning/training/callbacks:scenario_scoring_callback",
        "//nuplan/planning/training/callbacks/utils:scenario_scene_converter",
        "//nuplan/planning/training/data_augmentation:agent_dropout_augmentation",
        "//nuplan/planning/training/data_augmentation:kinematic_agent_augmentation",
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
    tags = [
        "cpu:20",
        "integration",
    ],
    deps = [
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/script/builders:data_augmentation_builder",
        "//nuplan/planning/script/builders:lr_scheduler_builder",
        "//nuplan/planning/script/builders:training_callback_builder",
        "//nuplan/planning/script/builders:training_metrics_builder",
        "//nuplan/planning/script/builders:worker_pool_builder",
        "//nuplan/planning/script/test:skeleton_test_train",
        "//nuplan/planning/training/callbacks:scenario_scoring_callback",
        "//nuplan/planning/training/callbacks/utils:scenario_scene_converter",
        "//nuplan/planning/training/data_augmentation:kinematic_agent_augmentation",
        "//nuplan/planning/training/data_loader:distributed_sampler_wrapper",
        "//nuplan/planning/training/data_loader:log_splitter",
        "//nuplan/planning/training/experiments:cache_metadata_entry",
        "//nuplan/planning/training/modeling/models:raster_model",
        "//nuplan/planning/training/modeling/models:urban_driver_open_loop_model",
        "//nuplan/planning/training/modeling/models:urban_driver_open_loop_model_utils",
        "//nuplan/planning/training/modeling/objectives:trajectory_weight_decay_imitation_objective",
        "//nuplan/planning/utils/multithreading:worker_pool",
    ],
)

py_test(
    name = "test_cache_nuplandb",
    size = "medium",
    srcs = ["test_cache_nuplandb.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/common/utils:s3_utils",
        "//nuplan/common/utils/test_utils:mock_s3_utils",
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/script/builders:data_augmentation_builder",
        "//nuplan/planning/script/builders:training_callback_builder",
        "//nuplan/planning/script/builders:training_metrics_builder",
        "//nuplan/planning/script/builders:worker_pool_builder",
        "//nuplan/planning/script/test:skeleton_test_train",
        "//nuplan/planning/training/callbacks/utils:scenario_scene_converter",
        "//nuplan/planning/training/data_loader:log_splitter",
        "//nuplan/planning/training/modeling/models:raster_model",
        requirement("hydra-core"),
    ],
)

py_test(
    name = "test_cache_mocked",
    size = "small",
    srcs = ["test_cache_mocked.py"],
    deps = [
        "//nuplan/planning/scenario_builder/test:mock_abstract_scenario_builder",
        "//nuplan/planning/script/builders:scenario_builder",
        "//nuplan/planning/script/test:skeleton_test_train",
        "//nuplan/planning/training/modeling/models:raster_model",
        "//nuplan/planning/training/preprocessing/feature_builders/test:mock_feature_utils",
        requirement_torch("torch"),
        requirement("hydra-core"),
    ],
)

py_test(
    name = "test_one_cycle_lr_scheduler",
    size = "enormous",
    srcs = ["test_one_cycle_lr_scheduler.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/planning/scenario_builder/nuplan_db:nuplan_scenario_builder",
        "//nuplan/planning/script/builders:data_augmentation_builder",
        "//nuplan/planning/script/builders:lr_scheduler_builder",
        "//nuplan/planning/script/builders:training_callback_builder",
        "//nuplan/planning/script/builders:training_metrics_builder",
        "//nuplan/planning/script/builders:worker_pool_builder",
        "//nuplan/planning/script/builders/utils:utils_config",
        "//nuplan/planning/script/test:skeleton_test_train",
        "//nuplan/planning/training/callbacks:scenario_scoring_callback",
        "//nuplan/planning/training/callbacks/utils:scenario_scene_converter",
        "//nuplan/planning/training/data_augmentation:kinematic_agent_augmentation",
        "//nuplan/planning/training/data_loader:distributed_sampler_wrapper",
        "//nuplan/planning/training/data_loader:log_splitter",
        "//nuplan/planning/training/experiments:cache_metadata_entry",
        "//nuplan/planning/training/modeling/models:lanegcn_utils",
        "//nuplan/planning/training/modeling/models:simple_vector_map_model",
        "//nuplan/planning/utils/multithreading:worker_pool",
    ],
)
