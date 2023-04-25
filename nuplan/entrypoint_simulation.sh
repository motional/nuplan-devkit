#!/bin/bash

set -uox pipefail

export PYTHONUNBUFFERED=1

if [[ -z "${NUPLAN_PLANNER}" ]]; then
  export NUPLAN_PLANNER="[remote_planner]"
fi


if [[ -n "${SCENARIO_FILTER_ID}" ]]; then
  # Override for API submission
  export CONTESTANT_ID="${APPLICANT_ID}"

  [ -d "/mnt/data" ] && cp -r /mnt/data/nuplan-v1.1/maps/* "${NUPLAN_MAPS_ROOT}"

  NUPLAN_CHALLENGE=`echo $NUPLAN_CHALLENGE | sed 's/\(.*\)_.*/\1/'`

  aws s3 cp s3://"${NUPLAN_SERVER_S3_ROOT_URL}"/"${CONTESTANT_ID}"/"${SUBMISSION_ID}"/submission_metadata.json /tmp/submission_metadata.json
  PHASE_NAME=`cat /tmp/submission_metadata.json | grep phase_name | sed 's/.*: "\(.*\)".*/\1/'`
  PHASE_SPLIT=`cat /tmp/submission_metadata.json | grep phase_split | sed 's/.*: "\(.*\).*"/\1/'`

  aws s3 cp s3://"${NUPLAN_SERVER_S3_ROOT_URL}"/"${S3_TOKEN_DIR}"/"${PHASE_NAME}"_scenarios/"${PHASE_NAME}"_tokens_"${SCENARIO_FILTER_ID}".txt /nuplan_devkit/
  aws s3 cp s3://"${NUPLAN_SERVER_S3_ROOT_URL}"/"${S3_TOKEN_DIR}"/"${PHASE_NAME}"_scenarios/"${PHASE_NAME}"_logs_"${SCENARIO_FILTER_ID}".txt /nuplan_devkit/
  while IFS= read -r line; do tokens+="\"$line\","; done < /nuplan_devkit/"${PHASE_NAME}"_tokens_"${SCENARIO_FILTER_ID}".txt
  tokens=$(echo $tokens | sed 's/\(.*\),/\1/')
  while IFS= read -r line; do logs+="${NUPLAN_DATA_ROOT_S3_URL}/splits/${PHASE_SPLIT}/$line.db,"; done < /nuplan_devkit/"${PHASE_NAME}"_logs_"${SCENARIO_FILTER_ID}".txt
  logs=$(echo $logs | sed 's/\(.*\),/\1/')
  conda run -n nuplan --no-capture-output python -u nuplan/planning/script/run_simulation.py \
         +simulation="${NUPLAN_CHALLENGE}" \
         planner="${NUPLAN_PLANNER}" \
         worker=sequential \
         scenario_builder=nuplan_challenge \
         scenario_builder.db_files="[${logs}]" \
         scenario_filter=nuplan_challenge_scenarios \
         scenario_filter.scenario_tokens="[${tokens}]" \
         contestant_id=\"${CONTESTANT_ID}\" \
         submission_id=\"${SUBMISSION_ID}\" \
         disable_callback_parallelization=true \
         main_callback.publisher_callback.s3_bucket="${NUPLAN_SERVER_S3_ROOT_URL}" \
         main_callback.publisher_callback.remote_prefix="[\"${CONTESTANT_ID}\", \"${SUBMISSION_ID}\"]"
else
    if [[ -z "${NUPLAN_CHALLENGE}" ]]; then
      export NUPLAN_CHALLENGE=open_loop_boxes
    fi
    conda run -n nuplan --no-capture-output python -u nuplan/planning/script/run_simulation.py \
         +simulation="${NUPLAN_CHALLENGE}" \
         planner="${NUPLAN_PLANNER}" \
         worker=sequential \
         scenario_builder=nuplan_challenge \
         scenario_filter=nuplan_challenge_scenarios \
         scenario_builder.db_files=/data/sets/nuplan/nuplan-v1.1/splits/mini/2021.07.16.20.45.29_veh-35_01095_01486.db \
         scenario_filter.limit_total_scenarios=1 \
         contestant_id="${CONTESTANT_ID}" \
         submission_id="${SUBMISSION_ID}"
fi
