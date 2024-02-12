#!/bin/bash

submit_job() {
  sub="$(sbatch "$@")"

  if [[ "$sub" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    exit 1
  fi
}

year=$1
flag_file="/users/zhanghui/GEDI/$year/done"
if [ ! -f "$config_file" ]; then
    echo "Continue downloading GEDI data for $year..."
    submit_job()
    exit 1
fi
# first job - no dependencies
id1=$(submit_job job1.sh)

# Two jobs that depend on the first job
id2=$(submit_job --dependency=afterany:$id1 job2.sh)
id3=$(submit_job --dependency=afterany:$id1 job3.sh)

# One job that depends on both the second and the third jobs
id4=$(submit_job  --dependency=afterany:$id2:$id3 job4.sh)
