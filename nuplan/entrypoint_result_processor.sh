#!/bin/bash

mkdir ~/.aws
touch ~/.aws/credentials
cat << EOF > ~/.aws/credentials
[default]
aws_access_key_id=$AWS_ACCESS_KEY_ID
aws_secret_access_key=$AWS_SECRET_ACCESS_KEY
EOF

conda run -n nuplan --no-capture-output python -u nuplan/planning/script/run_result_processor_aws.py \
       contestant_id="\"${APPLICANT_ID}\"" \
       submission_id="\"${SUBMISSION_ID}\""
