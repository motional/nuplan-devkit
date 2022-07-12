#!/bin/bash

if [[ -z "${NUPLAN_CHALLENGE}" ]]; then
  NUPLAN_CHALLENGE=open_loop_boxes
fi
if [[ -z "${NUPLAN_PLANNER}" ]]; then
  NUPLAN_PLANNER="[remote_planner]"
fi

python nuplan/planning/script/run_simulation.py \
       +simulation=$NUPLAN_CHALLENGE \
       planner=$NUPLAN_PLANNER \
       worker=sequential \
       scenario_filter.num_scenarios_per_type=2 \
       scenario_filter.limit_total_scenarios=2
