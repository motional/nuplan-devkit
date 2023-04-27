#!/bin/bash

set -uox pipefail

conda run -n nuplan --no-capture-output python -u nuplan/planning/script/run_result_processor_aws.py \
       contestant_id="\"${APPLICANT_ID}\"" \
       submission_id="\"${SUBMISSION_ID}\""
