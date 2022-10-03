# Final Metric Structure

This document describes how metrics are aggregated to generate the final score structure for comparison of planners performance on AV in Nuplan.

## Individual metric scores
We use a different set of metrics for evaluation of planners in challenge 1 and challenges 2 and 3. Each metric in the scenario score is assigned a value/score between 0-1 based on the planner performance in that scenario. A higher score shows a better performance according to that metric (See metrics scores' description in the table below and a more comprehensive description about metrics in general in [metric_description](https://github.com/motional/nuplan-devkit/blob/master/docs/metrics_description.md).

## Planner score in each scenario based on the proposed trajectory in challenge 1
In each scenario, selected metrics are aggregated to provide a score for the proposed trajectory. Currently, the aggregation function is a hybrid hierarchical-weighted average function of individual metric scores:
- The planner gets a zero score in a scenario if the miss rate is above the selected threshold,
- Otherwise, a weighted average of other metrics' scores is used as the score in that scenario.

|  Metric Name        | Metric Score      | Weight in Scenario Score     |
|--------------------|--------------------|---------------------------|
|Miss Rate Within Bound|0 if maximum displacement error between the planner proposed trajectory and expert trajectory is more than the `max_displacement_threshold` at the corresponding `comparison_horizon` in more than `max_miss_rate_threshold` of the time instances in the scenario, 1 otherwise.|NA/multiplying metric|
|Average Displacement Error (ADE)  Within Bound        |0 if the average of ADEs over the `comparison_horizon` is more than `max_average_l2_error_threshold`, 1  otherwise.|1|
|Final Displacement Error (FDE)  Within Bound        |0 if the average of ADEs over the `comparison_horizon` is more than `max_final_l2_error_threshold`, 1  otherwise.|1|
|Average heading error (AHE) Within Bound        |0 if the average of AHEs over the `comparison_horizon` is more than `max_average_heading_error_threshold`, 1  otherwise.|2|
|Final heading error (FHE) Within Bound       |0 if the average of FHEs over the `comparison_horizon` is more than `max_final_heading_error_threshold`, 1  otherwise.|2|



## Planner score in each scenario based on the driven trajectory in challenges 2 and 3

In each scenario, selected metrics are aggregated to provide a score for the driven trajectory. Currently, the aggregation function is a hybrid hierarchical-weighted average function of individual metric scores:

 - The planner gets a zero score for that driven trajectory/scenario if 
	- there is an at_fault collision with a vehicle or a VRU (pedestrian or bicyclist), or
	- there are multiple at_fault collisions with objects (e.g. a cone), or
	- there is a drivable_area violation,
	- ego drives into uncoming traffic more than 6 m, or 
	- ego is not making enough progress.
    
 - A weighted average of other metrics' scores is multiplied with 0.5 if there is one at_fault collision with an object (e.g. a cone), or if ego drives into uncoming traffic more than 2 m (but less than 6 m).
 - Otherwise, a weighted average of other metrics' scores is used as the score in that scenario.

Metrics scores, and how they are aggregated to compute the scenarios score is described in the following table. You can find metric thresholds/constants in [metric_description](https://github.com/motional/nuplan-devkit/blob/master/docs/metrics_description.md).

|  Metric Name        | Metric Score      | Weight in Scenario Score     |
|--------------------|--------------------|---------------------------|
|no_ego_at_fault_collisions |0 if there is an at-fault collision with a vehicle or a vru, or multiple at-fault collisions with objects, 0.5 if there's an at-fault collision with a single object, 1 otherwise.|NA/multiplying metric|
|drivable_area_compliance          |0 if at any instance the distance of a corner of ego's bounding box from the drivable area is more than `max_violation_threshold`, 1  otherwise.|NA/multiplying metric|
|driving_direction_compliance          |Score is 1 if during the previous `time_horizon` (for each time instance) ego has not been driving against the traffic flow more than  `driving_direction_compliance_threshold`, and 0 if it's been driving against the flow more than  `driving_direction_violation_threshold`, and 0.5 otherwise.|5|
|time_to_collision_within_bound   |0 if time_to_collision is less than `least_min_ttc` threshold, 1 otherwise. |5|
|speed_limit_compliance          | Score is ``max(0, 1 - (speed_violation_integral /(max_overspeed_value_threshold * total_scenario_duration)))``  <br> where ``speed_violation_integral`` is area under the speed_violation vs time graph, ``max_overspeed_value_threshold`` is the maximum acceptable over-speeding threshold, currently set at 2.23 m/s (equivalent to ~5mph), and ``total_scenario_duration`` is the scenario duration in seconds. <br> Therefore, score is 1 if there is no speed limit violation and approaches to 0 as the violation increases.|4|
|ego_progress_along_expert_route         |Score is 0 if `overall_ego_progress<0`, otherwise it is `min(1.0, max(overall_ego_progress, score_progress_threshold)/ max(overall_expert_progress, score_progress_threshold)` <br> where `overall_ego_progress` is ego's overall progress along the expert route, and <br> `overall_expert_progress` is expert overall progress along its route, and <br> `score_progress_threshold` is a small threshold.|5|
|ego_is_making_progress          | 0 if `ego_progress_along_expert_route` returns a value less than `min_progress_threshold`, 1 otherwise.  |NA/multiplying metric|
|Ego_is_comfortable  <small> <br> •  ego_jerk <br> •  ego_lat_acceleration <br> • ego_lon_acceleration <br> • ego_lon_jerk <br> •  ego_yaw_acceleration <br> • Ego_yaw_rate|0 if any of the comfort_metrics are not within comfort bounds/thresholds and 1 otherwise.|2|

## Planner score in each scenario type

nuPlan categorizes scenarios based on their types. The type is specified based on features of the scenario according to ego, agents, and the scene. A planner's score on one specific scenario type can be computed by averaging the scores of driven trajectories that belong to that scenario type. This score can be helpful in identifying the scenario types in which the planner does not perform well. This can be particularly helpful for ML planners. For instance if a planner gets an average score of 0.9 for 50 scenarios with scenario_type ON_INTERSECTION, and an average score of 0.6 for 60 scenarios of type ON_PICKUP_DROPOFF, we may want to consider improving the performance in scenarios where ego is on PUDO area.

## Planner final score

A planner is assigned a score between 0 and 1 based on its performance in multiple scenarios. A higher score shows a better performance. To assign a final score to a planner, it is run on N scenarios, and the driven trajectory scores in these N scenarios are averaged to score the planner.
